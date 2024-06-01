import torch
from typing import List, Dict, Optional

from torchmetrics import MeanMetric, CatMetric, SumMetric

import wandb
from sklearn.manifold import TSNE


class BaseMeter:
    def __init__(self, save_features: bool = False, device: torch.device = torch.device("cpu")) -> None:
        self.save_features = save_features
        self.device = device
        self.loss_meter = MeanMetric(nan_strategy="error").to(self.device)

        self.counter = SumMetric()

        if save_features:
            self.pre_features = CatMetric().to(device)
            self.post_features = CatMetric().to(device)

    def update(self, labels, loss, pre_features: Optional[torch.Tensor] = None, post_features: Optional[torch.Tensor] = None, *args, **kwargs) -> None:
        self.loss_meter.update(loss.detach())

        if self.save_features:
            if pre_features is not None:
                self.pre_features.update(pre_features.detach())
            if post_features is not None:
                self.post_features.update(post_features.detach())

        self.counter.update(labels.shape[0])

    def print_logs(self) -> List[str]:
        return [f"Loss: {self.loss_meter.compute():.4f}"]

    def plot_features(self, f, title):
        data = TSNE(2).fit_transform(f)
        table = wandb.Table(data=data, columns=['x', 'y'])
        return wandb.plot.scatter(table, "x", "y", title=title)

    def get_logs(self) -> Dict[str, str]:
        logs = {
            "loss": self.loss_meter.compute()
        }

        if self.save_features:
            logs.update({
                "pre_features": self.plot_features(self.pre_features.compute().cpu().numpy(), "Features before"),
                "post_features": self.plot_features(self.post_features.compute().cpu().numpy(), "Features after"),
            })

        return logs
