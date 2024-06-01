import torch
from torch import Tensor
from torchmetrics import Metric
from typing import Any, Callable, Dict, Union

from functools import reduce


class MultitaskAccuracy(Metric):
    def __init__(self, nlabels: int = 2, top_k: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.top_k = top_k
        self.nlabels = nlabels

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: tuple[torch.Tensor, ...], target: tuple[torch.Tensor, ...]):
        bs = target[0].shape[0]
        all_correct = torch.zeros(self.top_k, bs).type(torch.ByteTensor).to(self.device)

        for output, label in zip(preds, target):
            _, max_k_idx = output.topk(self.top_k, dim=1, largest=True, sorted=True)
            # Flip batch_size, class_count as .view doesn't work on non-contiguous
            max_k_idx = max_k_idx.t()
            correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
            all_correct.add_(correct_for_task)

        self.correct += torch.ge(all_correct.float().sum(0), self.nlabels).sum()
        self.total += all_correct.shape[1]

    def compute(self):
        return self.correct.float() / self.total


class ClassFilterWrapper(Metric):
    """
    Adapted from https://raw.githubusercontent.com/Lightning-AI/torchmetrics/v0.11.4/src/torchmetrics/wrappers/classwise.p

    """

    def __init__(self, metric: Metric, skip: Union[Tensor, tuple[Tensor, ...]] = None) -> None:
        super().__init__()
        if not isinstance(metric, Metric):
            raise ValueError(f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}")
        self.metric = metric
        self._update_count = 1
        self.skip = skip

    def _filter(self, *args: Any):
        if type(args[0]) != type(args[1]):
            raise ValueError("Expected arguments to share the same type")
        if type(args[0]) not in (tuple, torch.Tensor):
            raise ValueError("Expected arguments to be either a tuple of tensors or a tensor")
        
        if type(args[0]) == torch.Tensor:
            args = [arg[torch.isin(args[-1], self.skip.to(args[-1].device))] for arg in args]

        if type(args[0]) == tuple:
            filter = []
            for target, skip in zip(args[1], self.skip):
                skip = skip.to(target.device)
                filter.append(torch.isin(target, skip))

            filter = reduce(torch.bitwise_and, filter)
            preds, targets = [], []
            for p, t in zip(*args):
                preds.append(p[filter])
                targets.append(t[filter])

            args = [tuple(preds), tuple(targets)]

        return args

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.metric(*self._filter(*args), **kwargs)
    
    def update(self, *args: Any, **kwargs: Any) -> None:
        self.metric.update(*self._filter(*args), **kwargs)

    def compute(self) -> Dict[str, Tensor]:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()

    def _wrap_update(self, update: Callable) -> Callable:
        """Overwrite to do nothing."""
        return update

    def _wrap_compute(self, compute: Callable) -> Callable:
        """Overwrite to do nothing."""
        return compute


if __name__ == "__main__":
    acc_meter = MultitaskAccuracy()

    bs = 32768
    preds = (torch.randn((bs, 97)), torch.randn((bs, 300)))
    targets = (torch.randint(0, 97, (bs,)), torch.randint(0, 300, (bs,)))

    acc_meter(preds, targets)
    print(acc_meter.compute())

    bs = 512
    acc_meter = ClassFilterWrapper(MultitaskAccuracy(), skip=(torch.Tensor([0, 4, 3]), torch.Tensor([1,2,3,4,5])))
    preds = (torch.randn((bs, 5)), torch.randn((bs, 10)))
    targets = (torch.randint(0, 5, (bs,)), torch.randint(0, 10, (bs,)))

    acc_meter(preds, targets)
    print(acc_meter.compute())
