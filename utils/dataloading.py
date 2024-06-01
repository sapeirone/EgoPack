from torch_geometric.loader import DataLoader

import torch
import random
import numpy as np


class multiloader:
    def __init__(self, loaders, weights):
        self.loaders = loaders
        self.weights = weights
        # create an iterator for each loader
        self.iterators = [
            iter(loader) if loader is not None and weight > 0 else None
            for loader, weight in zip(self.loaders, self.weights)
        ]

        # mark absent iterator as already completed
        self.completed = [iterator is None for iterator in self.iterators]

    def __iter__(self):
        return self

    def __next__(self):
        data = []

        # iterate over all iterators
        for i in range(len(self.loaders)):
            if self.iterators[i] is None:
                data.append(None)
                continue

            try:
                # try loading from this iterator
                data.append(next(self.iterators[i]))  # type: ignore
            except StopIteration:
                # this iterator is exhausted -> mark it as completed
                self.completed[i] = True
                if all(self.completed):
                    # all iterators are exhausted -> we are done
                    raise StopIteration

                # some iterators still need to complete -> reset this iterator and keep looping
                self.iterators[i] = iter(self.loaders[i])
                data.append(next(self.iterators[i]))  # type: ignore

        return tuple(data)


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(dataset, batch_size, shuffle, num_workers, drop_last, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g
    )
