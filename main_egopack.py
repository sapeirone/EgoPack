import os.path as osp
import logging
import warnings

import hydra
import omegaconf

from graphone import build_graphone
from models.graphONE.graphONE import GraphONE
from models.tasks.pnr import PNRTask
from utils.meters import build_meter_for_dataset

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms.radius_graph import RadiusGraph
from torch_geometric.loader.dataloader import DataLoader

from torchmetrics.aggregation import MeanMetric

from data.base_dataset import BaseDataset
from validate import validate, validate_lta, validate_pnr

from models.tasks import RecognitionTask, OSCCTask, LTATask

from models.transforms.lta_temp_connectivity import LTATemporalConnectivity

import wandb

from utils.dataloading import build_dataloader, multiloader
from utils.wandb import format_wandb_run_name

from typing import Optional

from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


# TODO: temporary fix for torch geometric warning
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


def train_step_task(feat, batch, y,
                    primary_task, other_tasks,
                    graphone: GraphONE,
                    late_fusion: bool = True,):

    # forward features in the task backbone
    feat_primary = primary_task.forward_features(feat)  # features of the primary task

    secondary_tasks, _ = graphone.interact({task.name: task.forward_features(feat).detach() for task in other_tasks})

    if late_fusion:
        logits = primary_task.forward_logits(features=feat_primary, batch=batch, aux_features=secondary_tasks)
    else:
        feat = feat.max(1).values
        logits = primary_task.forward_logits(feat_primary, batch)

    return primary_task.compute_loss(logits, y)


def train(epoch, temporal_graph_model,
          graphone: GraphONE, late_fusion: bool,
          optimizer,
          # recognition
          ar_task: RecognitionTask, dl_ar, meter_ar,
          # OSCC
          oscc_task: OSCCTask, dl_oscc: DataLoader, meter_oscc,
          # long term anticipation
          lta_task: LTATask, dl_lta: DataLoader, meter_lta,
          # point of no return
          pnr_task: PNRTask, dl_pnr: DataLoader, meter_pnr,
          # task weights
          weight_ar=1.0, weight_oscc=1.0, weight_lta=1.0, weight_pnr=1.0,
          backprop_temporal_graph: bool = True, temporal_graph_train_mode: bool = False,
          device="cuda"):

    logger.debug(f"Starting training epoch {epoch}...")

    # Put all the models and tasks in training mode
    temporal_graph_model.train(temporal_graph_train_mode)

    tasks = [ar_task, lta_task, oscc_task, pnr_task]
    weights = [weight_ar, weight_lta, weight_oscc, weight_pnr]
    for task, weight in zip(tasks, weights):
        task.train(True)
        for name, p in task.named_parameters():
            p.requires_grad_(True)

    # Create a unified dataloader for all the tasks with length max(len(dl_i)) for i in )
    dataloader = multiloader(
        [dl_ar, dl_lta, dl_oscc, dl_pnr],
        [weight_ar, weight_lta, weight_oscc, weight_pnr]
    )

    if graphone is not None:
        graphone.train()

    completed_iterations = 0
    for data_ar, data_lta, data_oscc, data_pnr in tqdm(dataloader):
        optimizer.zero_grad()

        losses = []

        # # move data to the correct device
        data_ar: Optional[Data] = data_ar.to(device) if data_ar is not None else None
        data_oscc: Optional[Data] = data_oscc.to(device) if data_oscc is not None else None
        data_lta: Optional[Data] = data_lta.to(device) if data_lta is not None else None
        data_pnr: Optional[Data] = data_pnr.to(device) if data_pnr is not None else None

        with torch.set_grad_enabled(backprop_temporal_graph):
            feat_ar: Optional[torch.Tensor] = temporal_graph_model(data_ar) if data_ar is not None else None
            feat_oscc: Optional[torch.Tensor] = temporal_graph_model(data_oscc) if data_oscc is not None else None
            feat_lta: Optional[torch.Tensor] = temporal_graph_model(data_lta) if data_lta is not None else None
            feat_pnr: Optional[torch.Tensor] = temporal_graph_model(data_pnr) if data_pnr is not None else None

        # 1. Action Recognition (AR)
        if feat_ar is not None and data_ar is not None:
            loss = train_step_task(feat_ar, data_ar.batch, data_ar.y,
                                   ar_task, [lta_task, oscc_task, pnr_task],
                                   graphone, late_fusion)
            meter_ar.update(loss)
            losses.append(weight_ar * loss.mean())

        # 2. Object State Change Classification (OSCC)
        if feat_oscc is not None and data_oscc is not None:
            loss = train_step_task(feat_oscc, data_oscc.batch, data_oscc.y,
                                   oscc_task, [ar_task, lta_task, pnr_task],
                                   graphone, late_fusion)
            meter_oscc.update(loss)
            losses.append(weight_oscc * loss.mean())

        # 3. Long Term Anticipation (LTA)
        if feat_lta is not None and data_lta is not None:
            loss = train_step_task(feat_lta, data_lta.batch, data_lta.y,
                                   lta_task, [ar_task, oscc_task, pnr_task],
                                   graphone, late_fusion)
            meter_lta.update(loss)
            losses.append(weight_lta * loss.mean())

        # 4. Point of No Return
        if feat_pnr is not None and data_pnr is not None:
            loss = train_step_task(feat_pnr, data_pnr.batch, data_pnr.y,
                                   pnr_task, [ar_task, oscc_task, lta_task],
                                   graphone, late_fusion)
            meter_pnr.update(loss)
            losses.append(weight_pnr * loss.mean())

        # backpropagate the recognition and anticipation loss
        loss = torch.stack(losses).sum()

        loss.backward()
        optimizer.step()

        completed_iterations += 1

    logger.debug(f"Epoch {epoch} completed {completed_iterations} iterations.")


@hydra.main(config_path="configs/", config_name="defaults", version_base="1.3")
def main(cfg):

    wandb.init(entity="egorobots", project="ego-graph", name=format_wandb_run_name(cfg.wandb_name_pattern, cfg))
    wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.run.log_code(".")  # type: ignore

    if cfg.seed > 0:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
            torch.backends.cudnn.benchmark = False  # type: ignore
            # operations like scatter do not support deterministic execution
            # torch.use_deterministic_algorithms(True)

    if not cfg.enable_graphone:
        logging.warning("Invalid configuration. Aborting!")
        exit()

    logger.info("Starting MTL training process with the following configuration:")
    task_weights = {
        task: getattr(cfg, f'weight_{task}') if task in cfg.enabled_tasks else 0
        for task in ['ar', 'oscc', 'lta', 'pnr']
    }
    for task, weight in task_weights.items():
        logger.info(f" - Weight of {task} is {weight}")

    # Collect training datasets here
    dsets_train, dsets_val = [], []

    # 1. Initialize recognition dataset
    logger.info("")
    logger.info("Initializing recognition dataset...")
    dset_rec_train: BaseDataset = hydra.utils.instantiate(cfg.dataset_recognition, split="train", transform=RadiusGraph(r=cfg.k + 0.5, loop=False))
    dset_rec_val: BaseDataset = hydra.utils.instantiate(cfg.dataset_recognition, split=cfg.validation_split, transform=RadiusGraph(r=cfg.k + 0.5, loop=False))
    dsets_train.append(dset_rec_train)
    dsets_val.append(dset_rec_val)

    logger.info(f"Recognition features have size {dset_rec_train.features_size}.")
    logger.info(f"Size of the recognition train dataset is {len(dset_rec_train)}.")
    logger.info(f"Size of the recognition validation dataset is {len(dset_rec_val)}.")

    dl_rec_train = build_dataloader(dset_rec_train, cfg.batch_size, True, cfg.num_workers, True, cfg.seed)
    dl_rec_val = build_dataloader(dset_rec_val, cfg.batch_size, False, cfg.num_workers, False, cfg.seed)

    logger.info(f"Size of the recognition train dataloader is {len(dl_rec_train)}.")
    logger.info(f"Size of the recognition validation dataloader is {len(dl_rec_val)}.")

    # 2. Initialize oscc dataset
    logger.info("Initializing OSCC dataset...")
    dset_oscc_train = hydra.utils.instantiate(cfg.dataset_oscc, split="train", transform=RadiusGraph(r=cfg.k + 0.5, loop=False))
    dset_oscc_val = hydra.utils.instantiate(cfg.dataset_oscc, split=cfg.validation_split, transform=RadiusGraph(r=cfg.k + 0.5, loop=False))
    dsets_train.append(dset_oscc_train)
    dsets_val.append(dset_oscc_val)

    logger.info(f"OSCC features have size {dset_oscc_train.features_size}.")
    logger.info(f"Size of the OSCC train dataset is {len(dset_oscc_train)}.")
    logger.info(f"Size of the OSCC validation dataset is {len(dset_oscc_val)}.")

    dl_oscc_train: Optional[DataLoader] = build_dataloader(dset_oscc_train, cfg.batch_size, True, cfg.num_workers, True, cfg.seed)
    dl_oscc_val: Optional[DataLoader] = build_dataloader(dset_oscc_val, cfg.batch_size, False, cfg.num_workers, False, cfg.seed)

    logger.info(f"Size of the OSCC train dataloader is {len(dl_oscc_train)}.")
    logger.info(f"Size of the OSCC validation dataloader is {len(dl_oscc_val)}.")

    # 3. Initialize LTA dataset
    logger.info("Initializing LTA dataset...")
    dset_lta_train = hydra.utils.instantiate(cfg.dataset_lta, split="train", transform=LTATemporalConnectivity(r=cfg.k + 0.5, loop=False))
    dset_lta_val = hydra.utils.instantiate(cfg.dataset_lta, split=cfg.validation_split, transform=LTATemporalConnectivity(r=cfg.k + 0.5, loop=False))
    dsets_train.append(dset_lta_train)
    dsets_val.append(dset_lta_val)

    logger.info(f"LTA features have size {dset_lta_train.features_size}.")
    logger.info(f"Size of the LTA train dataset is {len(dset_lta_train)}.")
    logger.info(f"Size of the LTA validation dataset is {len(dset_lta_val)}.")

    dl_lta_train: Optional[DataLoader] = build_dataloader(dset_lta_train, cfg.batch_size, True, cfg.num_workers, True, cfg.seed)
    dl_lta_val: Optional[DataLoader] = build_dataloader(dset_lta_val, cfg.batch_size, False, cfg.num_workers, False, cfg.seed)

    logger.info(f"Size of the LTA train dataloader is {len(dl_lta_train)}.")
    logger.info(f"Size of the LTA validation dataloader is {len(dl_lta_val)}.")

    # 4. Initialize PNR dataset
    logger.info("")
    logger.info("Initializing PNR dataset...")
    dset_pnr_train: BaseDataset = hydra.utils.instantiate(cfg.dataset_pnr, split="train", transform=RadiusGraph(r=cfg.k + 0.5, loop=False))
    dset_pnr_val: BaseDataset = hydra.utils.instantiate(cfg.dataset_pnr, split=cfg.validation_split, transform=RadiusGraph(r=cfg.k + 0.5, loop=False))
    dsets_train.append(dset_pnr_train)
    dsets_val.append(dset_pnr_val)

    logger.info(f"PNR features have size {dset_pnr_train.features_size}.")
    logger.info(f"Size of the PNR train dataset is {len(dset_pnr_train)}.")
    logger.info(f"Size of the PNR validation dataset is {len(dset_pnr_val)}.")

    dl_pnr_train: DataLoader = build_dataloader(dset_pnr_train, cfg.batch_size, True, cfg.num_workers, True, seed=cfg.seed)
    dl_pnr_val: DataLoader = build_dataloader(dset_pnr_val, cfg.batch_size, False, cfg.num_workers, False, seed=cfg.seed)

    logger.info(f"Size of the PNR train dataloader is {len(dl_pnr_train)}.")
    logger.info(f"Size of the PNR validation dataloader is {len(dl_pnr_val)}.")

    # Some sanity checks on the input datasets
    # 1. we expect the same features size for both recognition and anticipation
    assert all(dset.features_size == dsets_train[0].features_size for dset in dsets_train), "Input features should have the same size for both all the tasks."
    assert all(dset.features_size == dsets_train[0].features_size for dset in dsets_val), "Input features should have the same size for both all the tasks."
    # 2. ...

    # instantiate the graph reasoning module using hydra
    model = hydra.utils.instantiate(cfg.model, input_size=dset_rec_train.features_size,
                                    num_segments=cfg.dataset_recognition.num_segments,
                                    _recursive_=False).to(cfg.device)

    # initalize the task models
    tasks = [
        recognition_task := RecognitionTask(cfg.model.hidden_size, cfg.model.hidden_size, heads=dset_rec_train.num_class_labels,
                                            dropout=cfg.task_dropout, head_dropout=cfg.task_head_dropout, aux_tasks=('oscc', 'lta', 'pnr')).to(cfg.device),
        oscc_task := OSCCTask(cfg.model.hidden_size, cfg.model.hidden_size,
                              dropout=cfg.task_dropout, head_dropout=cfg.task_head_dropout, aux_tasks=('ar', 'lta', 'pnr'), average_logits=True).to(cfg.device),
        lta_task := LTATask(cfg.model.hidden_size, cfg.model.hidden_size, heads=dset_lta_train.num_class_labels,
                            dropout=cfg.task_dropout, head_dropout=cfg.task_head_dropout, aux_tasks=('ar', 'oscc', 'pnr')).to(cfg.device),
        pnr_task := PNRTask(cfg.model.hidden_size, cfg.model.hidden_size,
                            dropout=cfg.task_dropout, head_dropout=cfg.task_head_dropout, aux_tasks=('ar', 'oscc', 'lta')).to(cfg.device)
    ]

    if cfg.resume_from:
        logger.info(f"Resuming training from checkpoint {cfg.resume_from}...")
        artifact: wandb.Artifact = wandb.run.use_artifact(cfg.resume_from, type='model')  # type: ignore
        datadir: str = artifact.download()
        checkpoint = torch.load(osp.join(datadir, 'checkpoint.pth'), map_location=cfg.device)
        # load checkpoints
        model.load_state_dict(checkpoint["temporal_graph"])
        recognition_task.load_state_dict(checkpoint["task/recognition"], strict=False)
        oscc_task.load_state_dict(checkpoint["task/oscc"], strict=False)
        lta_task.load_state_dict(checkpoint["task/lta"], strict=False)
        pnr_task.load_state_dict(checkpoint["task/pnr"], strict=False)

    # # Initialize the graphone
    dset_graphone = dset_rec_train
    graphone = build_graphone(model, recognition_task,
                                  [t for t in tasks if t.name in cfg.resume_from],
                                  dataloader=build_dataloader(dset_graphone, 256, False, cfg.num_workers, True, cfg.seed),
                                  device=cfg.device)

    # graphone = torch.load('graphone.pth')
    graphone = GraphONE(graphone, **cfg.graphone).to(cfg.device)

    # log gradients and parameters for both the temporal graph model and the tasks
    wandb.watch(model, log="all", log_freq=10)
    wandb.watch(graphone, log="all", log_freq=10)
    wandb.watch(recognition_task, log="all", log_freq=10)
    wandb.watch(oscc_task, log="all", log_freq=10)
    wandb.watch(lta_task, log="all", log_freq=10)
    wandb.watch(pnr_task, log="all", log_freq=10)

    # instantiate the optimizer using hydra
    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(cfg.optimizer, [
        *model.configure_optimizers(cfg.optimizer.weight_decay),
        *recognition_task.configure_optimizers(cfg.optimizer.weight_decay),
        *oscc_task.configure_optimizers(cfg.optimizer.weight_decay),
        *lta_task.configure_optimizers(cfg.optimizer.weight_decay),
        *pnr_task.configure_optimizers(cfg.optimizer.weight_decay),
        *graphone.parameters()
    ])

    # instantiate the lr scheduler using hydra
    scheduler: torch.optim.lr_scheduler.LRScheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer)
    if cfg.use_warmup:
        scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.LinearLR(optimizer, 0.001, 1, 5),
            scheduler,
        ])

    for epoch in range(1, cfg.num_epochs + 1):
        # Build meters for train
        meter_recognition_train = MeanMetric().to(cfg.device)
        meter_oscc_train = MeanMetric().to(cfg.device)
        meter_lta_train = MeanMetric().to(cfg.device)
        meter_pnr_train = MeanMetric().to(cfg.device)

        logger.info(f"Starting training process for epoch {epoch:3d} out of {cfg.num_epochs}...")

        train(epoch, model, graphone, cfg.late_fusion, optimizer,
              # recognition
              recognition_task, dl_rec_train, meter_recognition_train,
              # oscc
              oscc_task, dl_oscc_train, meter_oscc_train,
              # long term anticipation
              lta_task, dl_lta_train, meter_lta_train,
              # point of no return
              pnr_task, dl_pnr_train, meter_pnr_train,
              # task weights
              weight_ar=task_weights['ar'], weight_oscc=task_weights['oscc'],
              weight_lta=task_weights['lta'], weight_pnr=task_weights['pnr'],
              backprop_temporal_graph=cfg.backprop_temporal_graph, temporal_graph_train_mode=cfg.temporal_graph_train_mode,
              device=cfg.device)

        scheduler.step()

        logger.info("")
        logger.info("Rec loss: {:.4f}".format(meter_recognition_train.compute()))
        logger.info("OSCC loss: {:.4f}".format(meter_oscc_train.compute()))
        logger.info("LTA loss: {:.4f}".format(meter_lta_train.compute()))
        logger.info("PNR loss: {:.4f}".format(meter_pnr_train.compute()))
        logger.info("")

        wandb.log({
            "train/recognition/loss": meter_recognition_train.compute(),
            "train/oscc/loss": meter_oscc_train.compute(),
            "train/lta/loss": meter_lta_train.compute(),
            "train/pnr/loss": meter_pnr_train.compute(),
        }, step=epoch)

        logger.info(f"Starting validation process for epoch {epoch:3d} out of {cfg.num_epochs}...")

        # 1. Validate recognition
        if cfg.validate_all_tasks or task_weights['ar'] > 0:
            is_egopack = task_weights['ar'] > 0
            meter_recognition_val = build_meter_for_dataset(dset_rec_val, device=cfg.device)
            validate(epoch, model, dl_rec_val, meter_recognition_val,
                     recognition_task, 
                     [lta_task, oscc_task, pnr_task] if is_egopack else [], graphone if is_egopack else None, 
                     late_fusion=cfg.late_fusion, device=cfg.device)
            logger.info(" ## Recognition ## ")
            for log in meter_recognition_val.print_logs():
                logger.info(log)
            logger.info("")
            wandb.log({
                f"val/recognition/{key}": value
                for key, value
                in meter_recognition_val.get_logs().items()
            }, step=epoch)
            del meter_recognition_val

        # 3. Validate oscc
        if cfg.validate_all_tasks or task_weights['oscc'] > 0:
            is_egopack = task_weights['oscc'] > 0
            meter_oscc_val = build_meter_for_dataset(dset_oscc_val, device=cfg.device)
            validate(epoch, model, dl_oscc_val, meter_oscc_val,
                     oscc_task, 
                     [recognition_task, lta_task, pnr_task] if is_egopack else [], graphone if is_egopack else None, 
                     late_fusion=cfg.late_fusion, device=cfg.device)
            logger.info(" ## OSCC ## ")
            for log in meter_oscc_val.print_logs():
                logger.info(log)
            logger.info("")
            wandb.log({
                f"val/oscc/{key}": value
                for key, value
                in meter_oscc_val.get_logs().items()
            }, step=epoch)
            del meter_oscc_val

        # 4. Validate LTA
        if cfg.validate_all_tasks or task_weights['lta'] > 0:
            is_egopack = task_weights['lta'] > 0
            meter_lta_val = build_meter_for_dataset(dset_lta_val, device=cfg.device)
            validate_lta(model, dl_lta_val, meter_lta_val,
                         lta_task, 
                         [recognition_task, oscc_task, pnr_task] if is_egopack else [], graphone if is_egopack else None, 
                         late_fusion=cfg.late_fusion, device=cfg.device)
            logger.info(" ## LTA ## ")
            for log in meter_lta_val.print_logs():
                logger.info(log)
            logger.info("")
            wandb.log({
                f"val/lta/{key}": value
                for key, value
                in meter_lta_val.get_logs().items()
            }, step=epoch)
            del meter_lta_val

        # 5. Validate PNR
        if cfg.validate_all_tasks or task_weights['pnr'] > 0:
            is_egopack = task_weights['pnr'] > 0
            meter_pnr_val = build_meter_for_dataset(dset_pnr_val, device=cfg.device)
            validate_pnr(model, dl_pnr_val, meter_pnr_val, pnr_task, [recognition_task, lta_task, oscc_task],
                         graphone, late_fusion=cfg.late_fusion, device=cfg.device)
            logger.info(" ## PNR ## ")
            for log in meter_pnr_val.print_logs():
                logger.info(log)
            logger.info("")
            wandb.log({
                f"val/pnr/{key}": value
                for key, value
                in meter_pnr_val.get_logs().items()
            }, step=epoch)
            del meter_pnr_val

    if cfg.save_model:
        path = osp.join(wandb.run.dir, "checkpoint.pth")    # type: ignore
        logger.info(f"Saving model to {path}.")
        torch.save({
            "temporal_graph": model.state_dict(),
            "task/recognition": recognition_task.state_dict(),
            "task/oscc": oscc_task.state_dict() if oscc_task is not None else None,
            "task/lta": lta_task.state_dict() if lta_task is not None else None,
            "task/pnr": pnr_task.state_dict() if pnr_task is not None else None,
            "graphone": graphone.state_dict() if graphone is not None else None,
        }, path)
        artifact_name = f'{cfg.artifact_prefix}_' + '-'.join(sorted(task for task, weight in task_weights.items() if weight > 0))
        artifact = wandb.Artifact(name=artifact_name, type='model')
        artifact.add_file(local_path=osp.join(wandb.run.dir, f"checkpoint.pth"), name="checkpoint.pth")  # type: ignore
        artifact.save()


if __name__ == "__main__":
    logger.info("Starting...")

    main()

    wandb.finish()
