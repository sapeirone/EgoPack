import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

import torchvision

from models.tasks.task import ProjectionTask, TaskLiteral

from typing import Literal, Tuple, Mapping, Optional, Dict

import logging
logger = logging.getLogger(__name__)


class OSCCTask(ProjectionTask):
    """OSCC task."""

    def __init__(self, input_size: int, features_size: int,
                 dropout: float = 0, head_dropout: float = 0,
                 loss_func: Literal['ce', 'bce', 'focal'] = 'ce',
                 aux_tasks: Optional[Tuple[TaskLiteral, ...]] = None,
                 average_logits: bool = False):
        """OSCC Task.

        Parameters
        ----------
        input_size : int
            size of the input features
        features_size : int
            internal features size
        dropout : float, optional
            dropout in the projection task, by default 0
        head_dropout : float, optional
            _description_, by default 0
        loss_func : Literal['ce', 'bce', 'focal'], optional
            loss function for the oscc task, by default 'ce'
        aux_tasks : Optional[Tuple[TaskLiteral, ...]], optional
            auxiliary classification tasks, by default None
        average_logits : bool, optional
            if True logits of auxiliary tasks are averaged with the main task logits. Otherwise they are summed, by default False
        """
        super().__init__('oscc', input_size, features_size, dropout)

        logger.info(f"Initializing OSCC task with loss={loss_func}, dropout={dropout}, " +
                    f"head_dropout={head_dropout} and the following auxiliary classifiers: {aux_tasks}.")

        self.loss_func = loss_func

        self.classifier = self._build_classifier(head_dropout)

        if aux_tasks:
            self.aux_classifiers: Mapping[str, nn.Module] = nn.ModuleDict({
                task: self._build_classifier(head_dropout)
                for task in aux_tasks
            })  # type: ignore
            self.average_logits = average_logits

    def _build_classifier(self, head_dropout) -> nn.Module:
        return nn.Sequential(nn.Dropout(head_dropout), nn.Linear(self.features_size, 2))

    def configure_optimizers(self, _):
        return self.parameters()

    def forward_logits(self, features: torch.Tensor, batch: torch.Tensor,
                       aux_features: Optional[Dict[TaskLiteral, torch.Tensor]] = None, *args, **kwargs):
        # max pooling over each separate graph
        features = gnn.pool.global_max_pool(features, batch)
        # main classifier
        logits = self.classifier(features)

        if aux_features is not None:
            # auxiliary classifiers for the secondary tasks
            aux_logits = [self.forward_aux_logits(task_features, batch, task_name) for task_name, task_features in aux_features.items()]
            logits = torch.stack([logits, *aux_logits])
            # average or sum logits of the auxiliary tasks and the primary task
            logits = logits.mean(0) if self.average_logits else logits.sum(0)

        return logits

    def forward_aux_logits(self, features: torch.Tensor, batch: torch.Tensor, t: TaskLiteral = 'ar', *args, **kwargs):
        if not hasattr(self, 'aux_classifiers'):
            raise ValueError("OSCC task has no auxiliary classifiers.")

        features = gnn.pool.global_max_pool(features, batch)
        return self.aux_classifiers[t](features)

    def compute_loss(self, logits, targets):
        if self.loss_func == 'ce':
            return F.cross_entropy(logits, targets, ignore_index=-1, reduction='none', label_smoothing=0.1)
        elif self.loss_func == 'bce':
            targets = torch.nn.functional.one_hot(targets, 2).float()
            return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        elif self.loss_func == 'focal':
            targets = torch.nn.functional.one_hot(targets, 2).float()
            return torchvision.ops.focal_loss.sigmoid_focal_loss(logits, targets, alpha=0.5, gamma=2.0, reduction="none")
