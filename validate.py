import torch
from models.tasks.pnr import PNRTask
from utils.meters import BaseMeter
from torch_geometric.loader.dataloader import DataLoader

from models.tasks import RecognitionTask, OSCCTask, LTATask
from typing import List

import logging
logger = logging.getLogger(__name__)


@torch.no_grad()
def validate(
    epoch,
    temporal_graph_model: torch.nn.Module,
    dataloader: DataLoader,
    meter: BaseMeter,
    primary_task: RecognitionTask | OSCCTask,
    other_tasks: List[RecognitionTask | LTATask | OSCCTask | PNRTask] = [],
    graphone=None, late_fusion: bool = True,
    device: str = "cuda"
):
    temporal_graph_model.eval()

    # Set all tasks in evaluation mode
    for task in [primary_task, *other_tasks]:
        task.eval()
    # Set the graphone in evaluation mode, if it exists
    if graphone is not None:
        graphone.eval()

    for data in dataloader:
        data = data.to(device)

        feat = temporal_graph_model(data)  # Perform a single forward pass.
        feat_primary = primary_task.forward_features(feat)

        if graphone is not None:
            feat_secondary = {task.name: task.forward_features(feat) for task in other_tasks}
            feat_secondary, *_ = graphone.interact(feat_secondary)

            feat = torch.stack([feat_primary, *feat_secondary.values()], dim=1)

            if late_fusion:
                logits = primary_task.forward_logits(features=feat_primary, batch=data.batch, aux_features=feat_secondary)
            else:
                feat = torch.stack([feat_primary, *feat_secondary.values()], dim=1).max(1).values
                logits = primary_task.forward_logits(feat, data.batch)
        else:
            feat = feat_primary
            logits = primary_task.forward_logits(feat, data.batch)

        if len(data.x.shape) == 3:
            pre_features, post_features = data.x.mean(1), feat
        else:
            pre_features, post_features = data.x, feat

        loss = primary_task.compute_loss(logits, data.y).mean()
        meter.update(logits, data.y, loss, pre_features, post_features)


@torch.no_grad()
def validate_lta(
    temporal_graph_model: torch.nn.Module,
    dataloader: DataLoader,
    meter: BaseMeter,
    primary_task: LTATask,
    other_tasks: List[RecognitionTask | LTATask | OSCCTask | PNRTask] = [],
    graphone=None,
    late_fusion: bool = False,
    device: str = "cuda"
):
    temporal_graph_model.eval()

    # Set all tasks in evaluation mode
    for task in [primary_task, *other_tasks]:
        task.eval()
    # Set the graphone in evaluation mode, if it exists
    if graphone is not None:
        graphone.eval()

    for data in dataloader:
        data = data.to(device)

        feat = temporal_graph_model(data)
        feat_primary = primary_task.forward_features(feat)

        if graphone is not None:
            feat_secondary = {task.name: task.forward_features(feat) for task in other_tasks}
            feat_secondary, *_ = graphone.interact(feat_secondary)

            if late_fusion:
                logits = primary_task.forward_logits(features=feat_primary, batch=data.batch, aux_features=feat_secondary)
            else:
                feat = torch.stack([feat_primary, *feat_secondary.values()], dim=1).max(1).values
                logits = primary_task.forward_logits(feat, data.batch)
        else:
            feat = feat_primary
            logits = primary_task.forward_logits(feat, data.batch)

        predictions, logits = primary_task.generate_from_logits(logits)

        loss = primary_task.compute_loss(logits, data.y).mean()

        meter.update(logits, data.y, predictions, loss)


@torch.no_grad()
def validate_pnr(
    temporal_graph_model: torch.nn.Module,
    dataloader: DataLoader,
    meter: BaseMeter,
    primary_task: PNRTask,
    other_tasks: List[RecognitionTask | LTATask | OSCCTask | PNRTask] = [],
    graphone=None,
    late_fusion: bool = False,
    device: str = "cuda"
):
    temporal_graph_model.eval()

    # Set all tasks in evaluation mode
    for task in [primary_task, *other_tasks]:
        task.eval()
    # Set the graphone in evaluation mode, if it exists
    if graphone is not None:
        graphone.eval()

    for data in dataloader:
        data = data.to(device)

        feat = temporal_graph_model(data)
        feat_primary = primary_task.forward_features(feat)

        if graphone is not None:
            feat_secondary = {task.name: task.forward_features(feat) for task in other_tasks}
            feat_secondary, *_ = graphone.interact(feat_secondary)

            if late_fusion:
                logits = primary_task.forward_logits(features=feat_primary, batch=data.batch, aux_features=feat_secondary)
            else:
                feat = torch.stack([feat_primary, *feat_secondary.values()], dim=1).max(1).values
                logits = primary_task.forward_logits(feat)
        else:
            feat = feat_primary
            logits = primary_task.forward_logits(feat)

        loss = primary_task.compute_loss(logits, data.y.float())

        meter.update(logits, data.y, data.batch, data.start_frame, data.end_frame, data.pnr_frame, loss)
