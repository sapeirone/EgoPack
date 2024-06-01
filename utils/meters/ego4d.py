import wandb

import numpy as np

from typing import Dict, Union
import torch
from .base import BaseMeter

from torch_geometric.utils import unbatch

from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    ConfusionMatrix,
    MulticlassCalibrationError,
    BinaryRecall,
    BinaryAccuracy,
    BinaryAUROC
)
from torchmetrics import CatMetric

from torch_geometric.utils import scatter

from utils.confusion import Top2ConfusionMatrix

from utils.meters.utils import topk_recall_fast

from data.ego4d_fho import Ego4dRecognitionDataset, Ego4dAnticipationDataset, Ego4dLTADataset
from data.ego4d_oscc import Ego4dOSCCDataset

import editdistance


class Ego4dRecognitionMeter(BaseMeter):
    def __init__(self, dataset: Ego4dRecognitionDataset, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dataset = dataset

        # get the index of verb and nouns ground truth labels
        self.idx_verbs, self.idx_nouns = dataset.label_names.index("verbs"), dataset.label_names.index("nouns")
        self.verb_labels = dataset.class_labels[self.idx_verbs]
        self.noun_labels = dataset.class_labels[self.idx_nouns]

        # verb metrics
        self.verbs_top1 = MulticlassAccuracy(num_classes=len(self.verb_labels), top_k=1, average="micro", ignore_index=-1).to(self.device)
        self.verbs_top2 = MulticlassAccuracy(num_classes=len(self.verb_labels), top_k=2, average="micro", ignore_index=-1).to(self.device)
        self.verbs_top3 = MulticlassAccuracy(num_classes=len(self.verb_labels), top_k=3, average="micro", ignore_index=-1).to(self.device)
        self.verbs_top5 = MulticlassAccuracy(num_classes=len(self.verb_labels), top_k=5, average="micro", ignore_index=-1).to(self.device)
        self.verbs_mc = MulticlassAccuracy(num_classes=len(self.verb_labels), top_k=1, average="macro", ignore_index=-1).to(self.device)
        self.verbs_confusion = ConfusionMatrix("multiclass", num_classes=len(self.verb_labels), ignore_index=-1).to(self.device)
        self.verbs_calibration_error = MulticlassCalibrationError(len(self.verb_labels), ignore_index=-1).to(self.device)
        self.verbs_brier_score = MulticlassCalibrationError(len(self.verb_labels), n_bins=1, norm="l2", ignore_index=-1).to(self.device)
        self.verbs_top2_confusion = Top2ConfusionMatrix(len(self.verb_labels)).to(self.device)
        self.verbs_mc_top1 = MulticlassAccuracy(num_classes=len(self.verb_labels), top_k=1, average=None, ignore_index=-1).to(self.device)
        self.verbs_mc_top2 = MulticlassAccuracy(num_classes=len(self.verb_labels), top_k=2, average=None, ignore_index=-1).to(self.device)
        self.verbs_mc_top5 = MulticlassAccuracy(num_classes=len(self.verb_labels), top_k=5, average=None, ignore_index=-1).to(self.device)

        # noun metrics
        self.nouns_top1 = MulticlassAccuracy(num_classes=len(self.noun_labels), top_k=1, average="micro", ignore_index=-1).to(self.device)
        self.nouns_top2 = MulticlassAccuracy(num_classes=len(self.noun_labels), top_k=2, average="micro", ignore_index=-1).to(self.device)
        self.nouns_top3 = MulticlassAccuracy(num_classes=len(self.noun_labels), top_k=3, average="micro", ignore_index=-1).to(self.device)
        self.nouns_top5 = MulticlassAccuracy(num_classes=len(self.noun_labels), top_k=5, average="micro", ignore_index=-1).to(self.device)
        self.nouns_mc = MulticlassAccuracy(num_classes=len(self.noun_labels), top_k=1, average="macro", ignore_index=-1).to(self.device)
        self.nouns_confusion = ConfusionMatrix("multiclass", num_classes=len(self.noun_labels), ignore_index=-1).to(self.device)
        self.nouns_calibration_error = MulticlassCalibrationError(len(self.noun_labels), ignore_index=-1).to(self.device)
        self.nouns_brier_score = MulticlassCalibrationError(len(self.noun_labels), n_bins=1, norm="l2", ignore_index=-1).to(self.device)
        self.nouns_top2_confusion = Top2ConfusionMatrix(len(self.noun_labels)).to(self.device)
        self.nouns_mc_top1 = MulticlassAccuracy(num_classes=len(self.noun_labels), top_k=1, average=None, ignore_index=-1).to(self.device)
        self.nouns_mc_top2 = MulticlassAccuracy(num_classes=len(self.noun_labels), top_k=2, average=None, ignore_index=-1).to(self.device)
        self.nouns_mc_top5 = MulticlassAccuracy(num_classes=len(self.noun_labels), top_k=5, average=None, ignore_index=-1).to(self.device)

        self.mc_verb_loss = torch.zeros(len(self.verb_labels)).to(self.device)
        self.n_verbs = torch.zeros(len(self.verb_labels)).to(self.device)
        self.mc_noun_loss = torch.zeros(len(self.noun_labels)).to(self.device)
        self.n_nouns = torch.zeros(len(self.noun_labels)).to(self.device)

    def update(self, logits, labels, *args, **kwargs) -> None:
        super().update(labels, *args, **kwargs)

        if kwargs.get("verb_loss", None) is not None:
            verb_loss = kwargs.get("verb_loss", None)
            valids = labels[:, self.idx_verbs] != -1
            self.mc_verb_loss += scatter(verb_loss[valids], labels[valids, self.idx_verbs], dim=0, reduce="sum", dim_size=len(self.verb_labels))
            self.n_verbs += torch.bincount(labels[valids, self.idx_verbs], minlength=len(self.verb_labels))
        if kwargs.get("noun_loss", None) is not None:
            noun_loss = kwargs.get("noun_loss", None)
            valids = labels[:, self.idx_nouns] != -1
            self.mc_noun_loss += scatter(noun_loss[valids], labels[valids, self.idx_nouns], dim=0, reduce="sum", dim_size=len(self.noun_labels))
            self.n_nouns += torch.bincount(labels[valids, self.idx_nouns], minlength=len(self.noun_labels))

        # update metrics for verbs
        self.verbs_top1.update(logits[self.idx_verbs], labels[:, self.idx_verbs])
        self.verbs_top2.update(logits[self.idx_verbs], labels[:, self.idx_verbs])
        self.verbs_top3.update(logits[self.idx_verbs], labels[:, self.idx_verbs])
        self.verbs_top5.update(logits[self.idx_verbs], labels[:, self.idx_verbs])
        self.verbs_mc.update(logits[self.idx_verbs], labels[:, self.idx_verbs])
        self.verbs_confusion.update(logits[self.idx_verbs], labels[:, self.idx_verbs])
        self.verbs_calibration_error.update(logits[self.idx_verbs], labels[:, self.idx_verbs])
        self.verbs_brier_score.update(logits[self.idx_verbs], labels[:, self.idx_verbs])
        self.verbs_top2_confusion.update(logits[self.idx_verbs], labels[:, self.idx_verbs])
        self.verbs_mc_top1.update(logits[self.idx_verbs], labels[:, self.idx_verbs])
        self.verbs_mc_top2.update(logits[self.idx_verbs], labels[:, self.idx_verbs])
        self.verbs_mc_top5.update(logits[self.idx_verbs], labels[:, self.idx_verbs])

        # update metrics for nouns
        self.nouns_top1.update(logits[self.idx_nouns], labels[:, self.idx_nouns])
        self.nouns_top2.update(logits[self.idx_nouns], labels[:, self.idx_nouns])
        self.nouns_top3.update(logits[self.idx_nouns], labels[:, self.idx_nouns])
        self.nouns_top5.update(logits[self.idx_nouns], labels[:, self.idx_nouns])
        self.nouns_mc.update(logits[self.idx_nouns], labels[:, self.idx_nouns])
        self.nouns_confusion.update(logits[self.idx_nouns], labels[:, self.idx_nouns])
        self.nouns_calibration_error.update(logits[self.idx_nouns], labels[:, self.idx_nouns])
        self.nouns_brier_score.update(logits[self.idx_nouns], labels[:, self.idx_nouns])
        self.nouns_top2_confusion.update(logits[self.idx_nouns], labels[:, self.idx_nouns])
        self.nouns_mc_top1.update(logits[self.idx_nouns], labels[:, self.idx_nouns])
        self.nouns_mc_top2.update(logits[self.idx_nouns], labels[:, self.idx_nouns])
        self.nouns_mc_top5.update(logits[self.idx_nouns], labels[:, self.idx_nouns])

    def print_logs(self):
        mc_verb_loss = (self.mc_verb_loss / torch.clip(self.n_verbs, min=1))
        mc_noun_loss = (self.mc_noun_loss / torch.clip(self.n_nouns, min=1))

        return [
            f"Verbs Top-1: {self.verbs_top1.compute()*100:.2f}, Top-2: {self.verbs_top2.compute()*100:.2f}, Top-3: {self.verbs_top3.compute()*100:.2f}, Top-5: {self.verbs_top5.compute()*100:.2f}",
            f"Nouns Top-1: {self.nouns_top1.compute()*100:.2f}, Top-2: {self.nouns_top2.compute()*100:.2f}, Top-3: {self.nouns_top3.compute()*100:.2f}, Top-5: {self.nouns_top5.compute()*100:.2f}",
            f"Verbs Mean class: {self.verbs_mc.compute()*100:.2f} (Loss: {mc_verb_loss.mean():.2f} with std: {mc_verb_loss.std():.2f})",
            f"Nouns Mean class: {self.nouns_mc.compute()*100:.2f} (Loss: {mc_noun_loss.mean():.2f} with std: {mc_noun_loss.std():.2f})",
            f"Verbs Brier score: {self.verbs_brier_score.compute():.4f}",
            f"Nouns Brier score: {self.nouns_brier_score.compute():.4f}",
            *super().print_logs(),
        ]

    def get_logs(self, *args, **kwargs) -> Dict[str, Union[str, float, torch.Tensor, wandb.Table]]:
        # verbs top2 confusion
        verbs_top2_confusion = self.verbs_top2_confusion.compute().flatten()
        sorted_indices = torch.argsort(verbs_top2_confusion, descending=True)
        verbs_top2_confusion = [
            [
                self.verb_labels[idx // len(self.verb_labels)],
                self.verb_labels[idx % len(self.verb_labels)],
                verbs_top2_confusion[idx],
            ]
            for idx in sorted_indices[:25]
        ]

        # nouns top2 confusion
        nouns_top2_confusion = self.verbs_top2_confusion.compute().flatten()
        sorted_indices = torch.argsort(nouns_top2_confusion, descending=True)
        nouns_top2_confusion = [
            [
                self.noun_labels[idx // len(self.verb_labels)],
                self.noun_labels[idx % len(self.verb_labels)],
                nouns_top2_confusion[idx],
            ]
            for idx in sorted_indices[:25]
        ]

        return {
            # verbs metrics
            "verbs_top1": self.verbs_top1.compute(),
            "verbs_top2": self.verbs_top2.compute(),
            "verbs_top3": self.verbs_top3.compute(),
            "verbs_top5": self.verbs_top5.compute(),
            "verbs_mc": self.verbs_mc.compute(),
            "verbs_class_acc": wandb.Table(
                ["class", "top-1", "top-2", "top-5", "support"],
                list(
                    zip(
                        self.verb_labels,
                        self.verbs_mc_top1.compute(),
                        self.verbs_mc_top2.compute(),
                        self.verbs_mc_top5.compute(),
                        self.verbs_confusion.compute().sum(1).tolist(),
                    )
                ),
            ),
            "verbs_calibration_erorr": self.verbs_calibration_error.compute(),
            "verbs_brier_score": self.verbs_brier_score.compute(),
            "verbs_top2_confusion": wandb.Table(["ground_truth", "confused_with", "count"], verbs_top2_confusion),
            # nouns metrics
            "nouns_top1": self.nouns_top1.compute(),
            "nouns_top2": self.nouns_top2.compute(),
            "nouns_top3": self.nouns_top3.compute(),
            "nouns_top5": self.nouns_top5.compute(),
            "nouns_mc": self.nouns_mc.compute(),
            "nouns_class_acc": wandb.Table(
                ["class", "top-1", "top-2", "top-5", "support"],
                list(
                    zip(
                        self.noun_labels,
                        self.nouns_mc_top1.compute(),
                        self.nouns_mc_top2.compute(),
                        self.nouns_mc_top5.compute(),
                        self.nouns_confusion.compute().sum(1).tolist(),
                    )
                ),
            ),
            "nouns_calibration_erorr": self.nouns_calibration_error.compute(),
            "nouns_brier_score": self.nouns_brier_score.compute(),
            "nouns_top2_confusion": wandb.Table(["ground_truth", "confused_with", "count"], nouns_top2_confusion),
            **super().get_logs(*args, **kwargs),
        }


class Ego4dAnticipationMeter(BaseMeter):
    def __init__(self, dataset: Ego4dAnticipationDataset, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dataset = dataset

        # get the index of verb and nouns ground truth labels
        self.idx_verbs, self.idx_nouns = dataset.label_names.index("verbs"), dataset.label_names.index("nouns")
        self.verb_labels = dataset.class_labels[self.idx_verbs]
        self.noun_labels = dataset.class_labels[self.idx_nouns]

        # verbs metrics
        self.verbs_top1 = MulticlassAccuracy(num_classes=len(self.verb_labels), top_k=1, average="micro", ignore_index=-1).to(self.device)
        self.verbs_top2 = MulticlassAccuracy(num_classes=len(self.verb_labels), top_k=2, average="micro", ignore_index=-1).to(self.device)
        self.verbs_top3 = MulticlassAccuracy(num_classes=len(self.verb_labels), top_k=3, average="micro", ignore_index=-1).to(self.device)
        self.verbs_top5 = MulticlassAccuracy(num_classes=len(self.verb_labels), top_k=5, average="micro", ignore_index=-1).to(self.device)
        self.verbs_logits = CatMetric().to(self.device)
        self.verbs_ground_truths = CatMetric().to(self.device)

        # nouns metrics
        self.nouns_top1 = MulticlassAccuracy(num_classes=len(self.noun_labels), top_k=1, average="micro", ignore_index=-1).to(self.device)
        self.nouns_top2 = MulticlassAccuracy(num_classes=len(self.noun_labels), top_k=2, average="micro", ignore_index=-1).to(self.device)
        self.nouns_top3 = MulticlassAccuracy(num_classes=len(self.noun_labels), top_k=3, average="micro", ignore_index=-1).to(self.device)
        self.nouns_top5 = MulticlassAccuracy(num_classes=len(self.noun_labels), top_k=5, average="micro", ignore_index=-1).to(self.device)
        self.nouns_logits = CatMetric().to(self.device)
        self.nouns_ground_truths = CatMetric().to(self.device)

    def _get_valid_verbs_recall(self, k):
        recall = topk_recall_fast(self.verbs_logits.compute(), self.verbs_ground_truths.compute().long(), k)
        return float(recall)

    def _get_valid_nouns_recall(self, k):
        recall = topk_recall_fast(self.nouns_logits.compute(), self.nouns_ground_truths.compute().long(), k)
        return float(recall)

    @torch.no_grad()
    def update(self, logits, labels, *args, **kwargs) -> None:
        super().update(labels, *args, **kwargs)

        # update verbs metrics
        self.verbs_top1.update(logits[self.idx_verbs].detach(), labels[:, self.idx_verbs])
        self.verbs_top2.update(logits[self.idx_verbs].detach(), labels[:, self.idx_verbs])
        self.verbs_top3.update(logits[self.idx_verbs].detach(), labels[:, self.idx_verbs])
        self.verbs_top5.update(logits[self.idx_verbs].detach(), labels[:, self.idx_verbs])
        verb_logits, verb_gts = logits[self.idx_verbs].detach().cpu(), labels[:, self.idx_verbs].detach().cpu()
        self.verbs_logits.update(verb_logits[verb_gts != -1])
        self.verbs_ground_truths.update(verb_gts[verb_gts != -1])

        # update nouns metrics
        self.nouns_top1.update(logits[self.idx_nouns].detach(), labels[:, self.idx_nouns])
        self.nouns_top2.update(logits[self.idx_nouns].detach(), labels[:, self.idx_nouns])
        self.nouns_top3.update(logits[self.idx_nouns].detach(), labels[:, self.idx_nouns])
        self.nouns_top5.update(logits[self.idx_nouns].detach(), labels[:, self.idx_nouns])
        noun_logits, noun_gts = logits[self.idx_nouns].detach().cpu(), labels[:, self.idx_nouns].detach().cpu()
        self.nouns_logits.update(noun_logits[noun_gts != -1])
        self.nouns_ground_truths.update(noun_gts[noun_gts != -1])

    @torch.no_grad()
    def print_logs(self):
        return [
            # Verbs metrics
            f"Verbs Top-1: {self.verbs_top1.compute()*100:.2f}, Verbs Top-2: {self.verbs_top2.compute()*100:.2f}, Verbs Top-3: {self.verbs_top3.compute()*100:.2f}, Verbs Top-5: {self.verbs_top5.compute()*100:.2f}",
            f"Verbs Top-1 recall: {self._get_valid_verbs_recall(1)*100:.2f}, Verbs Top-2 recall: {self._get_valid_verbs_recall(2)*100:.2f}, Verbs Top-3 recall: {self._get_valid_verbs_recall(3)*100:.2f}, Verbs Top-5 recall: {self._get_valid_verbs_recall(5)*100:.2f}",
            # Nouns metrics
            f"Nouns Top-1: {self.nouns_top1.compute()*100:.2f}, Nouns Top-2: {self.nouns_top2.compute()*100:.2f}, Nouns Top-3: {self.nouns_top3.compute()*100:.2f}, Nouns Top-5: {self.nouns_top5.compute()*100:.2f}",
            f"Nouns Top-1 recall: {self._get_valid_nouns_recall(1)*100:.2f}, Nouns Top-2 recall: {self._get_valid_nouns_recall(2)*100:.2f}, Nouns Top-3 recall: {self._get_valid_nouns_recall(3)*100:.2f}, Nouns Top-5 recall: {self._get_valid_nouns_recall(5)*100:.2f}",
            *super().print_logs(),
        ]

    @torch.no_grad()
    def get_logs(self, *args, **kwargs) -> Dict[str, Union[str, float, torch.Tensor, wandb.Table]]:
        return {
            # verbs metrics
            "verbs_accuracy_top1": self.verbs_top1.compute(),
            "verbs_accuracy_top2": self.verbs_top2.compute(),
            "verbs_accuracy_top3": self.verbs_top3.compute(),
            "verbs_accuracy_top5": self.verbs_top5.compute(),
            "verbs_recall_top1": self._get_valid_verbs_recall(1),
            "verbs_recall_top2": self._get_valid_verbs_recall(2),
            "verbs_recall_top3": self._get_valid_verbs_recall(3),
            "verbs_recall_top5": self._get_valid_verbs_recall(5),
            # nouns metrics
            "nouns_accuracy_top1": self.nouns_top1.compute(),
            "nouns_accuracy_top2": self.nouns_top2.compute(),
            "nouns_accuracy_top3": self.nouns_top3.compute(),
            "nouns_accuracy_top5": self.nouns_top5.compute(),
            "nouns_recall_top1": self._get_valid_nouns_recall(1),
            "nouns_recall_top2": self._get_valid_nouns_recall(2),
            "nouns_recall_top3": self._get_valid_nouns_recall(3),
            "nouns_recall_top5": self._get_valid_nouns_recall(5),
            **super().get_logs(*args, **kwargs),
        }


class Ego4dOSCCMeter(BaseMeter):
    def __init__(self, dataset: Ego4dOSCCDataset, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dataset = dataset

        self.accuracy = MulticlassAccuracy(num_classes=2, average="micro", ignore_index=-1).to(self.device)

    @torch.no_grad()
    def update(self, logits, labels, *args, **kwargs) -> None:
        super().update(labels, *args, **kwargs)

        # update verbs metrics
        self.accuracy(logits, labels)

    @torch.no_grad()
    def print_logs(self):
        return [
            # Verbs metrics
            f"Accuracy: {self.accuracy.compute()*100:.2f}",
            *super().print_logs(),
        ]

    @torch.no_grad()
    def get_logs(self, *args, **kwargs) -> Dict[str, Union[str, float, torch.Tensor, wandb.Table]]:
        return {
            # verbs metrics
            "accuracy": self.accuracy.compute(),
            **super().get_logs(*args, **kwargs),
        }


class Ego4dPNRMeter(BaseMeter):
    def __init__(self, dataset: Ego4dOSCCDataset, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dataset = dataset

        self.accuracy = BinaryAccuracy().to(self.device)
        self.recall = BinaryRecall().to(self.device)
        self.auroc = BinaryAUROC().to(self.device)
        self.preds = CatMetric().to(self.device)

        self.loc_errors = []

    @torch.no_grad()
    def update(self, logits, labels, batch, start_frame, end_frame, pnr_frame, *args, **kwargs) -> None:
        super().update(labels, *args, **kwargs)

        logits = torch.sigmoid(logits)

        self.accuracy.update(logits, labels)
        self.recall.update(logits, labels)
        self.auroc.update(logits, labels)

        # update verbs metrics
        for preds, sf, ef, pf in zip(unbatch(logits, batch), start_frame, end_frame, pnr_frame):
            keyframe_loc_pred = torch.argmax(preds).item()
            keyframe_loc_pred_mapped = (ef - sf) / 16 * keyframe_loc_pred
            keyframe_loc_pred_mapped = keyframe_loc_pred_mapped.item()

            # absolute distance of the pnr frame from the start of the clip
            gt = pf.item() - sf.item()

            err_frame = abs(keyframe_loc_pred_mapped - gt)
            err_sec = err_frame/30
            self.loc_errors.append(err_sec)

    @torch.no_grad()
    def print_logs(self):
        return [
            # Verbs metrics
            # f"{Counter(self.preds.compute().tolist())}",
            f'accuracy: {self.accuracy.compute():.4f}',
            f'recall: {self.recall.compute():.4f}',
            f"auroc: {self.auroc.compute():.4f}",
            f"localization_error: {np.mean(np.array(self.loc_errors)):.4f}",
            *super().print_logs(),
        ]

    @torch.no_grad()
    def get_logs(self, *args, **kwargs) -> Dict[str, Union[str, float, torch.Tensor, wandb.Table]]:
        return {
            # verbs metrics
            'accuracy': self.accuracy.compute(),
            'recall': self.recall.compute(),
            'auroc': self.auroc.compute(),
            'localization_error': float(np.mean(np.array(self.loc_errors))),
            **super().get_logs(*args, **kwargs),
        }


class Ego4dLTAMeter(BaseMeter):
    def __init__(self, dataset: Ego4dLTADataset, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dataset = dataset

        # get the index of verb and nouns ground truth labels
        self.idx_verbs, self.idx_nouns = dataset.label_names.index("verbs"), dataset.label_names.index("nouns")
        self.verb_labels = dataset.class_labels[self.idx_verbs]
        self.noun_labels = dataset.class_labels[self.idx_nouns]

        self.verbs_edit_distance = MeanMetric()
        self.nouns_edit_distance = MeanMetric()

        # verbs metrics
        self.verbs_top1 = MulticlassAccuracy(num_classes=len(self.verb_labels), top_k=1, average="micro", ignore_index=-1).to(self.device)
        self.nouns_top1 = MulticlassAccuracy(num_classes=len(self.noun_labels), top_k=1, average="micro", ignore_index=-1).to(self.device)

    def _edit_distance(self, preds, labels) -> torch.Tensor:
        """
        Damerau–Levenshtein edit distance from: https://github.com/gfairchild/pyxDamerauLevenshtein
        Lowest among K predictions
        """
        N, Z, K = preds.shape
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        dists = []
        for n in range(N):
            dist = min([editdistance.eval(preds[n, :, k], labels[n])/Z for k in range(K)])
            dists.append(dist)
        return torch.from_numpy(np.array(dists))

    @torch.no_grad()
    def update(self, logits, labels, predictions, *args, **kwargs) -> None:
        super().update(labels, *args, **kwargs)

        self.verbs_top1.update(logits[self.idx_verbs][labels[:, self.idx_verbs] >= 0], labels[labels[:, self.idx_verbs] >= 0, self.idx_verbs])
        self.nouns_top1.update(logits[self.idx_nouns][labels[:, self.idx_nouns] >= 0], labels[labels[:, self.idx_nouns] >= 0, self.idx_nouns])

        # update verbs metrics
        self.verbs_edit_distance.update(self._edit_distance(predictions[self.idx_verbs].reshape((-1, 22, 5))[:, 2:], labels[:, self.idx_verbs].reshape((-1, 22))[:, 2:]))
        self.nouns_edit_distance.update(self._edit_distance(predictions[self.idx_nouns].reshape((-1, 22, 5))[:, 2:], labels[:, self.idx_nouns].reshape((-1, 22))[:, 2:]))

    @torch.no_grad()
    def print_logs(self):
        return [
            # Verbs metrics
            f"verbs_ed: {self.verbs_edit_distance.compute():.4f}",
            f"nouns_ed: {self.nouns_edit_distance.compute():.4f}",
            f"verbs_top1: {self.verbs_top1.compute():.4f}",
            f"nouns_top1: {self.nouns_top1.compute():.4f}",
            *super().print_logs(),
        ]

    @torch.no_grad()
    def get_logs(self, *args, **kwargs) -> Dict[str, Union[str, float, torch.Tensor, wandb.Table]]:
        return {
            # verbs metrics
            "verbs_ed": self.verbs_edit_distance.compute(),
            "nouns_ed": self.nouns_edit_distance.compute(),
            **super().get_logs(*args, **kwargs),
        }
