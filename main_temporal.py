# 
# Main code for the temporal graph reasoning model training.
#
# This code is used to train the temporal models


import logging
import warnings

import hydra
import omegaconf

from criterion.wrapper import MetricSelectorWrapper
from utils.meters import build_meter_for_dataset

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms.radius_graph import RadiusGraph
from torch_geometric.loader.dataloader import DataLoader

from torchmetrics.aggregation import MeanMetric

from data.base_dataset import BaseDataset
from validate import validate, validate_lta, validate_pnr

from models.tasks import RecognitionTask, OSCCTask, LTATask, PNRTask

import wandb

import os.path as osp

from utils.dataloading import build_dataloader, multiloader
from utils.wandb import format_wandb_run_name

from typing import Optional

from models.transforms.lta_temp_connectivity import LTATemporalConnectivity


logger = logging.getLogger(__name__)


# TODO: temporary fix for torch geometric warning
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


def train(epoch, model, optimizer,
          # recognition
          ar_task: RecognitionTask, dl_ar, criterion_ar, meter_ar,
          # OSCC
          oscc_task: OSCCTask, dl_oscc: DataLoader, criterion_oscc, meter_oscc,
          # long term anticipation
          lta_task: LTATask, dl_lta: DataLoader, criterion_lta, meter_lta,
          # point of no return
          pnr_task: PNRTask, dl_pnr: DataLoader, criterion_pnr, meter_pnr,
          # task weights
          weight_ar=1.0, weight_oscc=1.0, weight_lta=1.0, weight_pnr=1.0,
          device="cuda"):
    logger.debug(f"Starting training epoch {epoch}...")

    # Put all the models and tasks in training mode
    model.train()

    for task in [ar_task, oscc_task, lta_task, pnr_task]:
        task.train()

    # Create a unified dataloader for all the tasks with length max(len(dl_i)) for i in )
    dataloader = multiloader(
        [dl_ar, dl_lta, dl_oscc, dl_pnr],
        [weight_ar, weight_lta, weight_oscc, weight_pnr]
    )

    completed_iterations = 0
    for data_ar, data_lta, data_oscc, data_pnr in dataloader:
        optimizer.zero_grad()

        losses = []

        # move data to the correct device
        data_ar: Optional[Data] = data_ar.to(device, non_blocking=True) if data_ar is not None else None
        data_oscc: Optional[Data] = data_oscc.to(device, non_blocking=True) if data_oscc is not None else None
        data_lta: Optional[Data] = data_lta.to(device, non_blocking=True) if data_lta is not None else None
        data_pnr: Optional[Data] = data_pnr.to(device, non_blocking=True) if data_pnr is not None else None

        feat_ar_temp: Optional[torch.Tensor] = model(data_ar) if data_ar is not None else None
        feat_oscc_temp: Optional[torch.Tensor] = model(data_oscc) if data_oscc is not None else None
        feat_lta_temp: Optional[torch.Tensor] = model(data_lta) if data_lta is not None else None
        feat_pnr_temp: Optional[torch.Tensor] = model(data_pnr) if data_pnr is not None else None

        # 1. Action Recognition (AR)
        if feat_ar_temp is not None:
            feat_ar = ar_task.forward_features(feat_ar_temp)
            logits_ar = ar_task.forward_logits(feat_ar)
            loss_ar = criterion_ar(logits_ar, data_ar.y)
            meter_ar.update(loss_ar)

            losses.append(weight_ar * loss_ar.mean())

        # 2. Long Term Anticipation (LTA)
        if feat_lta_temp is not None:
            feat_lta = lta_task.forward_features(feat_lta_temp)
            logits_lta = lta_task.forward_logits(feat_lta)
            loss_lta = criterion_lta(logits_lta, data_lta.y)
            meter_lta.update(loss_lta)

            losses.append(weight_lta * loss_lta.mean())

        # 3. Object State Change Classification (OSCC)
        if feat_oscc_temp is not None:
            feat_oscc = oscc_task.forward_features(feat_oscc_temp)
            logits_oscc = oscc_task.forward_logits(feat_oscc, data_oscc.batch)
            loss_oscc = criterion_oscc(logits_oscc, data_oscc.y)
            meter_oscc.update(loss_oscc)

            losses.append(weight_oscc * loss_oscc.mean())

        # 4. Point of No Return (PNR)
        if feat_pnr_temp is not None:
            feat_pnr = pnr_task.forward_features(feat_pnr_temp)
            logits_pnr = pnr_task.forward_logits(feat_pnr)
            loss_pnr = criterion_pnr(logits_pnr, data_pnr.y.float())
            meter_pnr.update(loss_pnr)

            losses.append(weight_pnr * loss_pnr.mean())

        torch.stack(losses).sum().backward()

        optimizer.step()

        completed_iterations += 1

    logger.info(f"Epoch {epoch} completed {completed_iterations} iterations.")


@hydra.main(config_path="configs/", config_name="defaults", version_base="1.3")
def main(cfg):

    wandb.init(entity="egorobots", project="ego-graph", name=format_wandb_run_name(cfg.wandb_name_pattern, cfg))
    wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.run.log_code(".")

    if cfg.seed > 0:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
            torch.backends.cudnn.benchmark = False

    logger.info("Starting training process with the following configuration:")
    task_weights = {
        task: getattr(cfg, f'weight_{task}') if task in cfg.enabled_tasks else 0
        for task in ['ar', 'oscc', 'lta', 'pnr']
    }
    for task, weight in task_weights.items():
        logger.info(f" - Weight of {task} is {weight}")

    artifact_name = f'{cfg.artifact_prefix}_' + '-'.join(sorted(task for task, weight in task_weights.items() if weight > 0))
    logger.info(f"This run will provide artifact {artifact_name}.")

    # Collect training datasets here
    dsets_train, dsets_val = [], []

    # 1. Initialize action recognition dataset
    logger.info("")
    logger.info("Initializing AR dataset...")
    dset_ar_train: BaseDataset = hydra.utils.instantiate(cfg.dataset_recognition, split="train", transform=RadiusGraph(r=cfg.k + 0.5, loop=False))
    dset_ar_val: BaseDataset = hydra.utils.instantiate(cfg.dataset_recognition, split=cfg.validation_split, transform=RadiusGraph(r=cfg.k + 0.5, loop=False))
    dsets_train.append(dset_ar_train)
    dsets_val.append(dset_ar_val)

    logger.info(f"AR features have size {dset_ar_train.features_size}.")
    logger.info(f"Size of the AR train dataset is {len(dset_ar_train)}.")
    logger.info(f"Size of the AR validation dataset is {len(dset_ar_val)}.")

    dl_ar_train = build_dataloader(dset_ar_train, cfg.batch_size, True, cfg.num_workers, True, seed=cfg.seed)
    dl_ar_val = build_dataloader(dset_ar_val, cfg.batch_size, False, cfg.num_workers, False, seed=cfg.seed)

    logger.info(f"Size of the AR train dataloader is {len(dl_ar_train)}.")
    logger.info(f"Size of the AR validation dataloader is {len(dl_ar_val)}.")

    avg_ar_length = torch.tensor([data.x.shape[0] for data in dset_ar_train]).float().mean() * cfg.batch_size
    logger.info("Average length of AR videos is {:.2f}.".format(avg_ar_length))

    # 2. Initialize OSCC dataset
    logger.info("")
    logger.info("Initializing OSCC dataset...")
    dset_oscc_train: BaseDataset = hydra.utils.instantiate(cfg.dataset_oscc, split="train", transform=RadiusGraph(r=cfg.k + 0.5, loop=False))
    dset_oscc_val: BaseDataset = hydra.utils.instantiate(cfg.dataset_oscc, split=cfg.validation_split, transform=RadiusGraph(r=cfg.k + 0.5, loop=False))
    dsets_train.append(dset_oscc_train)
    dsets_val.append(dset_oscc_val)

    logger.info(f"OSCC features have size {dset_oscc_train.features_size}.")
    logger.info(f"Size of the OSCC train dataset is {len(dset_oscc_train)}.")
    logger.info(f"Size of the OSCC validation dataset is {len(dset_oscc_val)}.")

    dl_oscc_train: DataLoader = build_dataloader(dset_oscc_train, cfg.batch_size, True, cfg.num_workers, True, seed=cfg.seed)
    dl_oscc_val: DataLoader = build_dataloader(dset_oscc_val, cfg.batch_size, False, cfg.num_workers, False, seed=cfg.seed)

    logger.info(f"Size of the OSCC train dataloader is {len(dl_oscc_train)}.")
    logger.info(f"Size of the OSCC validation dataloader is {len(dl_oscc_val)}.")

    # 3. Initialize LTA dataset
    logger.info("")
    logger.info("Initializing LTA dataset...")
    dset_lta_train: BaseDataset = hydra.utils.instantiate(cfg.dataset_lta, split="train", transform=LTATemporalConnectivity(r=cfg.k + 0.5, loop=False))
    dset_lta_val: BaseDataset = hydra.utils.instantiate(cfg.dataset_lta, split=cfg.validation_split, transform=LTATemporalConnectivity(r=cfg.k + 0.5, loop=False))
    dsets_train.append(dset_lta_train)
    dsets_val.append(dset_lta_val)

    logger.info(f"LTA features have size {dset_lta_train.features_size}.")
    logger.info(f"Size of the LTA train dataset is {len(dset_lta_train)}.")
    logger.info(f"Size of the LTA validation dataset is {len(dset_lta_val)}.")

    dl_lta_train: DataLoader = build_dataloader(dset_lta_train, cfg.batch_size, True, cfg.num_workers, True, seed=cfg.seed)
    dl_lta_val: DataLoader = build_dataloader(dset_lta_val, cfg.batch_size, False, cfg.num_workers, False, seed=cfg.seed)

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

    logger.info("")

    # instantiate the graph reasoning module using hydra
    model = hydra.utils.instantiate(cfg.model, input_size=dset_ar_train.features_size,
                                    num_segments=cfg.dataset_recognition.num_segments,
                                    _recursive_=False).to(cfg.device)

    # initalize the task models
    ar_task = RecognitionTask(cfg.model.hidden_size, cfg.model.hidden_size, heads=dset_ar_train.num_class_labels, dropout=cfg.task_dropout, head_dropout=cfg.task_head_dropout).to(cfg.device)
    oscc_task = OSCCTask(cfg.model.hidden_size, cfg.oscc_feat_size, dropout=cfg.task_dropout, head_dropout=cfg.task_head_dropout, loss_func=cfg.oscc_loss).to(cfg.device)
    lta_task = LTATask(cfg.model.hidden_size, cfg.model.hidden_size, heads=dset_lta_train.num_class_labels, dropout=cfg.task_dropout, head_dropout=cfg.task_head_dropout).to(cfg.device)
    pnr_task = PNRTask(cfg.model.hidden_size, cfg.model.hidden_size, dropout=cfg.task_dropout, head_dropout=cfg.task_head_dropout).to(cfg.device)

    logger.info("")

    # log gradients and parameters for both the temporal graph model and the tasks
    wandb.watch(model, log="all", log_freq=10)

    # instantiate the optimizer using hydra
    optimizer = hydra.utils.instantiate(cfg.optimizer, [
        *model.configure_optimizers(cfg.optimizer.weight_decay),
        *ar_task.configure_optimizers(cfg.optimizer.weight_decay),
        *oscc_task.configure_optimizers(cfg.optimizer.weight_decay),
        *lta_task.configure_optimizers(cfg.optimizer.weight_decay),
        *pnr_task.configure_optimizers(cfg.optimizer.weight_decay),
    ])

    # instantiate the lr scheduler using hydra
    scheduler: torch.optim.lr_scheduler.LRScheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer)
    if cfg.use_warmup:
        scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.LinearLR(optimizer, 0.001, 1, 5),
            scheduler,
        ])

    # For each sample, the model outputs as many samples as the number of labels
    # the dataset. However, the model may be trained on a subset of these labels
    # or on a joint combination of multiple class labels.
    # The objective of the MetricSelectorWrapper class is apply the criterion on
    # the logits according to the train_metric argument.
    criterion_ar: torch.nn.Module = MetricSelectorWrapper(
        torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-1).to(cfg.device),
        dataset=dset_ar_train,
    )
    
    criterion_oscc: nn.Module = nn.CrossEntropyLoss(reduction="none", ignore_index=-1).to(cfg.device)

    criterion_lta: torch.nn.Module = MetricSelectorWrapper(
        torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-1).to(cfg.device),
        dataset=dset_lta_train,
    )

    criterion_pnr: nn.Module = nn.BCEWithLogitsLoss(reduction="none").to(cfg.device)

    for epoch in range(1, cfg.num_epochs + 1):
        # Build meters for train
        meter_ar_train = MeanMetric().to(cfg.device)
        meter_oscc_train = MeanMetric().to(cfg.device)
        meter_lta_train = MeanMetric().to(cfg.device)
        meter_pnr_train = MeanMetric().to(cfg.device)

        logger.info(f"Starting training process for epoch {epoch:3d} out of {cfg.num_epochs}...")

        train(epoch, model, optimizer,
              # recognition
              ar_task, dl_ar_train, criterion_ar, meter_ar_train,
              # oscc
              oscc_task, dl_oscc_train, criterion_oscc, meter_oscc_train,
              # long term anticipation
              lta_task, dl_lta_train, criterion_lta, meter_lta_train,
              # Point of No Return
              pnr_task, dl_pnr_train, criterion_pnr, meter_pnr_train,
              # task weights
              weight_ar=task_weights['ar'], weight_oscc=task_weights['oscc'], 
              weight_lta=task_weights['lta'], weight_pnr=task_weights['pnr'],
              device=cfg.device)

        logger.info("")
        logger.info("Rec loss: {:.4f}".format(meter_ar_train.compute()))
        logger.info("OSCC loss: {:.4f}".format(meter_oscc_train.compute()))
        logger.info("LTA loss: {:.4f}".format(meter_lta_train.compute()))
        logger.info("PNR loss: {:.4f}".format(meter_pnr_train.compute()))
        logger.info("")

        wandb.log({
            "train/recognition/loss": meter_ar_train.compute(),
            "train/oscc/loss": meter_oscc_train.compute(),
            "train/lta/loss": meter_lta_train.compute(),
            "train/pnr/loss": meter_pnr_train.compute(),
        }, step=epoch)

        del meter_ar_train, meter_oscc_train, meter_lta_train, meter_pnr_train

        scheduler.step()
        logger.info("Learning rate updated to {:.6f}.".format(scheduler.get_last_lr()[0]))
        
        if epoch < (cfg.num_epochs - 5):
            continue

        logger.info(f"Starting validation process for epoch {epoch:3d} out of {cfg.num_epochs}...")

        # 1. Validate recognition
        if task_weights['ar'] > 0:
            meter_ar_val = build_meter_for_dataset(dset_ar_val, device=cfg.device)
            validate(epoch, model, dl_ar_val, meter_ar_val, ar_task, device=cfg.device)
            logger.info(" ## Recognition ## ")
            for log in meter_ar_val.print_logs():
                logger.info(log)
            logger.info("")
            wandb.log({
                f"val/recognition/{key}": value
                for key, value
                in meter_ar_val.get_logs().items()
            }, step=epoch)
            del meter_ar_val

        # 2. Validate LTA
        if task_weights['lta'] > 0:
            meter_lta_val = build_meter_for_dataset(dset_lta_val, device=cfg.device)
            validate_lta(model, dl_lta_val, meter_lta_val, lta_task, device=cfg.device)
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

        # 3. Validate OSCC
        if task_weights['oscc'] > 0:
            meter_oscc_val = build_meter_for_dataset(dset_oscc_val, device=cfg.device)
            validate(epoch, model, dl_oscc_val, meter_oscc_val, oscc_task, device=cfg.device)
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

        # 4. Validate PNR
        if task_weights['pnr'] > 0:
            meter_pnr_val = build_meter_for_dataset(dset_pnr_val, device=cfg.device)
            validate_pnr(model, dl_pnr_val, meter_pnr_val, pnr_task, device=cfg.device)
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
        path = osp.join(wandb.run.dir, "checkpoint.pth")
        logger.info(f"Saving model to {path}.")
        torch.save({
            "temporal_graph": model.state_dict(),
            "task/recognition": ar_task.state_dict(),
            "task/oscc": oscc_task.state_dict() if oscc_task is not None else None,
            "task/lta": lta_task.state_dict() if lta_task is not None else None,
            "task/pnr": pnr_task.state_dict() if pnr_task is not None else None,
            "epoch": epoch,
        }, path)
        artifact = wandb.Artifact(name=artifact_name, type='model')
        artifact.add_file(local_path=osp.join(wandb.run.dir, f"checkpoint.pth"), name="checkpoint.pth")
        artifact.save()

    if cfg.save_artifacts:
        for name, value in model.get_artifacts().items():
            torch.save(value, osp.join(wandb.run.dir, f"{name}.pth"))
            artifact = wandb.Artifact(name=name, type='model')
            artifact.add_file(local_path=osp.join(wandb.run.dir, f"{name}.pth"), name=name)
            artifact.save()


if __name__ == "__main__":
    logger.info("Starting...")

    main()
