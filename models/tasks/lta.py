import torch
import torch.nn as nn

from torch.distributions import Categorical
from typing import Tuple, Mapping, Optional, Dict

from models.tasks.task import ProjectionTask, TaskLiteral


class LTATask(ProjectionTask):
    """Multihead long-term anticipation task."""

    def __init__(self, input_size: int, features_size: int, heads: Tuple[int, ...],
                 dropout: float = 0, head_dropout: int = 0,
                 aux_tasks: Optional[Tuple[TaskLiteral, ...]] = None,
                 average_logits: bool = False):
        super().__init__('lta', input_size, features_size, dropout)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

        self.classifiers: nn.ModuleList = self._build_classifier(head_dropout, heads)

        if aux_tasks:
            self.aux_classifiers: Mapping[str, nn.ModuleList] = nn.ModuleDict({
                task: self._build_classifier(head_dropout, heads)
                for task in aux_tasks
            })  # type: ignore
            self.average_logits = average_logits

    def _build_classifier(self, head_dropout, heads) -> nn.ModuleList:
        return nn.ModuleList([
            nn.Sequential(nn.Dropout(head_dropout), nn.Linear(self.features_size, head))
            for head in heads
        ])

    def configure_optimizers(self, _):
        return self.parameters()

    def forward_logits(self, features: torch.Tensor, batch: Optional[torch.Tensor] = None,
                       aux_features: Optional[Dict[TaskLiteral, torch.Tensor]] = None, *args, **kwargs):
        # main classifier
        logits = tuple(_cls(features) for _cls in self.classifiers)

        if aux_features is not None:
            # auxiliary classifiers for the secondary tasks
            aux_logits = [
                self.forward_aux_logits(task_features, task_name) for task_name, task_features in aux_features.items()
            ]

            new_logits = []
            for primary_head_logits, aux_head_logits in zip(logits, zip(*aux_logits)):
                head_logits = torch.stack([primary_head_logits, *aux_head_logits])
                # average or sum logits of the auxiliary tasks and the primary task
                head_logits = head_logits.mean(0) if self.average_logits else head_logits.sum(0)
                new_logits.append(head_logits)
            logits = tuple(new_logits)

        return logits

    def forward_aux_logits(self, features: torch.Tensor, t: TaskLiteral = 'ar', *args, **kwargs):
        return tuple(_cls(features) for _cls in self.aux_classifiers[t])

    def generate_from_logits(self, logits: Tuple[torch.Tensor], K=5, *args, **kwargs):
        predictions = []
        for head_logits in logits:
            preds_dist = Categorical(logits=head_logits)
            preds = [preds_dist.sample() for _ in range(K)]
            head_x = torch.stack(preds, dim=1)
            predictions.append(head_x)

        return predictions, logits

    def compute_loss(self, logits: Tuple[torch.Tensor], targets: torch.Tensor):
        return torch.stack([self.loss_fn(l, t) for l, t in zip(logits, targets.unbind(1))]).sum(0)
