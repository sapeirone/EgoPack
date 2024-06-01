import torch

import torchmetrics
from torchmetrics import Metric
from torchmetrics.aggregation import CatMetric
from typing import Any


class Top2ConfusionMatrix(Metric):

    def __init__(self, num_classes, ignore_index=-1, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.predictions = CatMetric()
        self.ground_truth_labels = CatMetric()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = preds[target != self.ignore_index], target[target != self.ignore_index]

        # Computing top-1 and top-2 predictions
        top1_predictions = torch.argmax(preds, dim=1)
        _, top2_predictions = torch.topk(preds, k=2, dim=1)

        # Filtering samples where top-1 prediction is incorrect but top-2 prediction is correct
        incorrect_top1_mask = top1_predictions != target
        correct_top2_mask = top2_predictions[:, 1] == target
        filtered_samples_mask = incorrect_top1_mask & correct_top2_mask

        # Getting the filtered predicted labels and ground truth labels
        self.predictions.update(top1_predictions[filtered_samples_mask])
        self.ground_truth_labels.update(target[filtered_samples_mask])

    def compute(self):
        predictions = self.predictions.compute()
        ground_truth_labels = self.ground_truth_labels.compute()

        if not isinstance(predictions, torch.Tensor):
            predictions = torch.Tensor(predictions)
            ground_truth_labels = torch.Tensor(ground_truth_labels)

        return torchmetrics.functional.confusion_matrix(
            predictions,
            ground_truth_labels,
            'multiclass', num_classes=self.num_classes, ignore_index=-1
        )
