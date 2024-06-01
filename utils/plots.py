import seaborn as sns
import matplotlib.pyplot as plt

import wandb


def plot_confusion_matrix(confusion, normalize=True):
    if normalize:
        confusion = confusion / (1 + confusion.sum(1, keepdims=True))

    fig = plt.figure(dpi=300)
    sns.heatmap(confusion)
    return fig


def generate_confusion_table(confusion, labels, normalize=True):
    data = []

    if normalize:
        confusion = confusion / confusion.sum(1, keepdims=True)

    confusion = confusion.long()
    for i, label_actual in enumerate(labels):
        for j, label_pred in enumerate(labels):
            if confusion[i][j] > 0:
                data.append((f"{label_actual} --> {label_pred}", confusion[i][j]))

    return wandb.Table(columns=["Actual --> Predicted", "nPredictions"], data=data)