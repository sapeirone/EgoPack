import torch

from typing import Tuple

from data.base_dataset import BaseDataset

import logging
logger = logging.getLogger(__name__)


class MetricSelectorWrapper(torch.nn.Module):
    """Selectively apply a loss criterion on a subset of the model predictions.

    This module may be used to wrap a loss criterion. If the dataset has a joint
    label, e.g. the action label in EPIC-Kitchens-100, and the joint_label_training
    flag is set to True, the criterion is applied only on the joint label. If the flag
    is set to False, the criterion is applied on all the labels, except the joint label.
    If the dataset does not support a joint label, the criterion is applied on all the
    labels.
    """

    def __init__(self, criterion: torch.nn.Module, dataset: BaseDataset, joint_label_training: bool = False) -> None:
        """Initialize MetricSelectorWrapper.

        Parameters
        ----------
        criterion : torch.nn.Module
            wrapped loss criterion
        dataset : VideoDataset
            dataset
        joint_label_training : bool
            whether the loss should be applied on the joint label or not
        """
        super().__init__()

        if not dataset.has_joint_label and joint_label_training:
            logger.warn("The flag join_labels is set to True but the dataset has no joint label")
            joint_label_training = False

        self.criterion = criterion
        self.dataset = dataset
        self.joint_label = joint_label_training

    def forward(self, logits: Tuple[torch.Tensor], ground_truths: torch.Tensor) -> torch.Tensor:
        """Apply the criterion on the model logits.

        Parameters
        ----------
        logits : Tuple[torch.Tensor]
            model outputs, each of shape batch_size x num_classes
        ground_truths : torch.Tensor
            ground truth labels with shape batch_size x num_labels

        Returns
        -------
        torch.Tensor
            loss computed for each sample

        Raises
        ------
        ValueError
            if there is a mismatch between the number of predictions and the number of ground truth labels
        """
        if len(logits) != ground_truths.shape[1]:
            raise ValueError("The number of predictions must match the number of ground truth labels")

        losses = []

        if self.dataset.has_joint_label:
            # dataset has a joint label

            if self.joint_label:
                # apply the criterion on the joint label
                losses.append(self.criterion(logits[-1], ground_truths[:, -1]))
            else:
                # apply the criterion on all the labels, except the joint labels
                losses += [self.criterion(logits[i], ground_truths[:, i]) for i in range(self.dataset.num_labels - 1)]
        else:
            # apply criterion on all labels
            losses += [self.criterion(logits[i], ground_truths[:, i]) for i in range(self.dataset.num_labels)]

        return torch.stack(losses).sum(0)
