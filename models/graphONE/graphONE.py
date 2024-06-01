import torch
import torch.nn as nn
from torch_geometric.utils import add_remaining_self_loops
import torch_geometric.nn as gnn

import logging
from typing import Literal, Dict, Tuple, List


logger = logging.getLogger(__name__)


class GraphONE(nn.Module):

    def __init__(self,
                 graphone: Dict[str, torch.Tensor],
                 features_size: int = 1024,
                 hidden_size: int = 1024,
                 freeze: bool = True,
                 # interaction
                 k: int = 8,
                 depth: int = 3,
                 distance_func: Literal['l2', 'cosine'] = 'cosine',
                 residual: bool = False,
                 mix_strategy: Literal['mean', 'max', 'transformer'] = 'max',
                 update_edges_interval: int = 1,  # 0 means edges are computed just once, n > 0 means edges are updated every n iterations
                 share_params: bool = False,
                 *args, **kwargs) -> None:
        super().__init__()

        self.feature_size = features_size

        # interaction parameters
        self.k = k
        self.distance_func = distance_func
        self.residual = residual
        self.mix_strategy = mix_strategy

        self.update_edges_interval = update_edges_interval
        self.share_cnn_params = share_params

        logger.info(f"GraphONE initialized with {len(graphone)} tasks using depth={depth} and K={k}.")
        if not freeze:
            logger.warning("GraphONE initialized with trainable prototypes.")

        self.task_labels = sorted(graphone.keys())
        self.embeddings = nn.ModuleDict({
            task: nn.Embedding.from_pretrained(graphone[task], freeze=freeze) for task in self.task_labels
        })

        self.conv_stages = dict()
        self.depth = depth
        for task in self.task_labels:
            conv_layers = []
            norm_layers = []
            act_layers = []
            proj_layers = []

            for i in range(depth):
                conv_layers.append(gnn.SAGEConv(self.feature_size, hidden_size, bias=False, project=False, aggr='max'))
                norm_layers.append(nn.LayerNorm(hidden_size))
                act_layers.append(nn.ReLU())
                proj_layers.append(gnn.Linear(hidden_size, self.feature_size))

            self.conv_stages[task] = nn.ModuleList([
                gnn.Sequential('x, edge_index, weights', [
                    (conv, 'x, edge_index -> x'),
                    (norm, 'x -> x'),
                    (act, 'x -> x'),
                    (proj, 'x -> x'),
                ])
                for conv, norm, act, proj in zip(conv_layers, norm_layers, act_layers, proj_layers)
            ])
        self.conv_stages = nn.ModuleDict(self.conv_stages)

    def interact(self, features: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
        output: Dict[str, torch.Tensor] = dict()
        closest_nodes: Dict[str, List[torch.Tensor]] = dict()

        for task in features.keys():
            output[task], closest_nodes[task] = self.__task_interaction(task,
                                                                           features[task], self.embeddings[task].weight,
                                                                           features[task], self.embeddings[task].weight)

        return output, closest_nodes

    def __task_interaction(self, task, features, graphone, features_match, graphone_match) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        conv_stages: nn.ModuleList = self.conv_stages[task]  # type: ignore

        node_assignments: List[torch.Tensor] = []

        # as seen in https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DeepGCNLayer.html
        edges = None
        for depth, conv in enumerate(conv_stages):
            # NOTE: features at this stage have normalized to a zero mean and unit variance distribution
            # graphone is built from features normalized in the same way
            # Therefore, at least at the first iteration the two appear to be comparable.
            # What happens at the following iterations?
            # graphone and features are unlikely to have again the same distribution as the input features, unless
            # there is a residual path also on the graphone allowing to compare apples to apples.

            online_edges, *_, closest_nodes = self.__compute_edges(features_match, graphone_match)
            node_assignments.append(closest_nodes)

            if edges is None or (self.update_edges_interval and depth % self.update_edges_interval == 0):
                edges = online_edges

            graph = torch.cat([graphone, features], dim=0)
            edges, _ = add_remaining_self_loops(edges, num_nodes=graph.shape[0])
            graph = conv(graph, edges, 0.5 * torch.ones(edges.shape[1], device=edges.device))

            if self.residual:
                features = graph[-features.shape[0]:] + features
            else:
                features = graph[-features.shape[0]:]

        return features, node_assignments

    @torch.no_grad()
    def __compute_edges(self, features, graphone):

        K = graphone.shape[0]
        B = features.shape[0]

        if self.distance_func == 'l2':
            distances = cdist(features, graphone) / 4096
        elif self.distance_func == 'cosine':
            # cossim is 1 when the distance is 0, so we need to invert it (fix)
            distances = cos_dissimilarity(features, graphone)
        else:
            raise ValueError(f"Unknown distance function: {self.distance_func}")

        closest_nodes = distances.argsort(dim=-1, descending=False)[:, :self.k]
        edges = torch.stack([closest_nodes.flatten(), torch.arange(K, K + B, device=graphone.device).repeat_interleave(closest_nodes.shape[1])])

        distances = distances.sort(dim=-1, descending=False).values[:, :self.k]

        weights = (1/distances).softmax(dim=-1)
        entropy = -(weights * weights.log()).sum(dim=-1).mean()

        return edges, weights.flatten(), entropy, closest_nodes[:, 0]


def cdist(g1, g2):
    return torch.cdist(g1, g2, p=2, compute_mode='donot_use_mm_for_euclid_dist')


def cos_dissimilarity(g1, g2):
    g1 = g1 / g1.norm(dim=1, keepdim=True)
    g2 = g2 / g2.norm(dim=1, keepdim=True)
    return (1 - torch.mm(g1, g2.T))


def cossim(g1, g2):
    g1 = g1 / g1.norm(dim=1, keepdim=True)
    g2 = g2 / g2.norm(dim=1, keepdim=True)
    return torch.mm(g1, g2.T)


if __name__ == "__main__":
    device = 'cuda'
    initial_graphone = torch.rand((10_000, 4096), device=device, requires_grad=False)

    #print(gnn.summary(graphone_pooling, features))
    print(torch.cuda.max_memory_allocated() / 1024 ** 3)
