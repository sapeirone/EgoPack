import torch
import torch.nn as nn

from models.tasks.task import ProjectionTask, TaskLiteral

from typing import Tuple, Mapping, Optional, Dict

import logging
logger = logging.getLogger(__name__)


class PNRTask(ProjectionTask):
    """Point of No Return task."""

    def __init__(self, input_size: int, features_size: int,
                 dropout: float = 0, head_dropout: int = 0,
                 aux_tasks: Optional[Tuple[TaskLiteral, ...]] = None,
                 average_logits: bool = False):
        """PNR Task.

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
        aux_tasks : Optional[Tuple[TaskLiteral, ...]], optional
            auxiliary classification tasks, by default None
        average_logits : bool, optional
            if True logits of auxiliary tasks are averaged with the main task logits. Otherwise they are summed, by default False
        """
        super().__init__('pnr', input_size, features_size, dropout)

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        self.classifier = self._build_classifier(head_dropout)

        if aux_tasks:
            self.aux_classifiers: Mapping[str, nn.Module] = nn.ModuleDict({
                task: self._build_classifier(head_dropout)
                for task in aux_tasks
            })  # type: ignore
            self.average_logits = average_logits

    def _build_classifier(self, head_dropout) -> nn.Module:
        return nn.Sequential(nn.Dropout(head_dropout), nn.Linear(self.features_size, 1))

    def configure_optimizers(self, _):
        return self.parameters()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        features = self.net(x)
        return self.classifier(features).squeeze(), features

    def forward_features(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.net(x)

    def forward_logits(self, features: torch.Tensor,
                       aux_features: Optional[Dict[TaskLiteral, torch.Tensor]] = None, *args, **kwargs):
        # main classifier
        logits = self.classifier(features).squeeze()

        if aux_features is not None:
            # auxiliary classifiers for the secondary tasks
            aux_logits = [self.forward_aux_logits(task_features, task_name) for task_name, task_features in aux_features.items()]
            logits = torch.stack([logits.unsqueeze(1), *aux_logits])
            # average or sum logits of the auxiliary tasks and the primary task
            logits = logits.mean(0) if self.average_logits else logits.sum(0)

        return logits.squeeze()

    def forward_aux_logits(self, features: torch.Tensor, t: TaskLiteral = 'ar', *args, **kwargs):
        if not hasattr(self, 'aux_classifiers'):
            raise ValueError("PNR task has no auxiliary classifiers.")

        return self.aux_classifiers[t](features)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        return self.loss_fn(logits, targets.float())
