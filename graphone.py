import torch
from torch_geometric.data import Data
from torch_geometric.utils import scatter

import logging
from typing import List

from models.tasks.task import ProjectionTask
from models.tasks.recognition import RecognitionTask

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def build_graphone(model, ar_task: RecognitionTask, tasks: List[ProjectionTask],
                   dataloader,
                   device="cuda"):
    logger.debug("Building graphONE...")

    # put all the models and tasks in eval mode
    model.eval()
    for task in tasks:
        task.eval()

    logger.info("Building graphONE from tasks: %s", ", ".join([task.name for task in tasks]))

    feat_size = ar_task.net[-1].out_features
    n_classes = tuple(classifier[-1].out_features for classifier in ar_task.classifiers)  # type: ignore
    size = n_classes[0] * n_classes[1]

    # keep track of the number of samples seen for each class
    all_labels = []
    # graphone is initialized in float64 to avoid overflow errors during the generation process
    graphone = {task.name: torch.zeros((size, feat_size), dtype=torch.float64, device=device) for task in tasks}

    for data in tqdm(dataloader):
        data: Data = data.to(device)
        # common temporal modelling of the input segments
        feat: torch.Tensor = model(data)

        feat = feat[data.y[:, 0] != -1]
        y = data.y[data.y[:, 0] != -1]

        # collect the different point of views of the different tasks
        for task in tasks:
            task_feat = task.forward_features(feat)

            # hard assignment using labels (if available)
            labels = y[:, 0] * n_classes[1] + y[:, 1]
            all_labels.append(labels)
            graphone[task.name] = graphone[task.name] + scatter(task_feat, labels, dim_size=size, reduce="sum")

    bincount = torch.cat(all_labels).bincount(minlength=size).float()

    # remove (verb, noun) combinations that were never seen
    graphone = {
        task: (task_graph[bincount > 0] / bincount[bincount > 0, None]).float()
        for task, task_graph in graphone.items()
    }

    return graphone
