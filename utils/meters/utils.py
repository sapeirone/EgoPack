import numpy as np
import torch
import torchmetrics


def topk_accuracy(scores, labels, ks, selected_class=None):
    """Computes TOP-K accuracies for different values of k
    Args:
        scores: numpy nd array, shape = (instance_count, label_count)
        labels: numpy nd array, shape = (instance_count,)
        ks: tuple of integers
    Returns:
        list of float: TOP-K accuracy for each k in ks
    """
    if selected_class is not None:
        idx = labels == selected_class
        scores = scores[idx]
        labels = labels[idx]
    rankings = scores.argsort()[:, ::-1]
    maxk = np.max(ks)  # trim to max k to avoid extra computation

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    return [tp[:, :k].max(1).mean() for k in ks]


def topk_recall(scores, labels, k=5, classes=None):
    unique = np.unique(labels)
    if classes is None:
        classes = unique
    else:
        classes = np.intersect1d(classes, unique)
    recalls = 0

    for c in classes:
        recalls += topk_accuracy(scores, labels, ks=(k,), selected_class=c)[0]
    return recalls / len(classes)

@torch.no_grad()
def topk_recall_fast(scores, labels, k=5):
    n_classes = scores.shape[1]
    unique_labels = torch.unique(labels)

    metric = torchmetrics.functional.classification.multiclass_recall(scores, labels, num_classes=n_classes, top_k=k, average=None)
    return torch.mean(metric[torch.isin(torch.arange(len(metric), device=scores.device), unique_labels)])


if __name__ == "__main__":
    n_samples = 1024
    n_classes = 2048

    res_1, res_2 = [], []
    for i in range(20):
        scores = torch.rand(n_samples, n_classes).cuda()
        labels = torch.randint(0, n_classes, (n_samples,)).long().cuda()

        res_1.append(topk_recall(scores.cpu().numpy(), labels.cpu().numpy(), k=5))
        res_2.append(topk_recall_fast(scores, labels, k=5).cpu())

    print(torch.allclose(torch.from_numpy(np.array(res_1)).float(), torch.stack(res_2).float()))
