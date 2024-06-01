import json
import os
import os.path as osp
import logging
from typing import Literal
from collections import namedtuple

import random
from einops import rearrange

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data
from data.base_dataset import BaseFrameDataset

from tqdm.auto import tqdm

from data.ego4d import (
    Ego4dBackbones,
    __features_window_sizes__,
    __features_strides__,
    __features_sizes__
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


Ego4dOSCCPNREntry = namedtuple("Ego4dOSCCEntry", [
    "video_uid", "unique_uid",
    "start_frame", "end_frame",
    "start_sec", "end_sec",
    "state_change", "pnr_frame"
])

# As stated here (https://ego4d-data.org/docs/data/features/):
#  "These features are extracted from the canonical videos. Canonical videos are all 30FPS."
__FPS__ = 30


class Ego4dOSCCDataset(BaseFrameDataset, Dataset):
    """Ego4D datase for the Object State Change Classification (OSCC) and Point of No Return (PNR) tasks.

    This dataset can support the following tasks:
     - Object State Change Classification (OSCC)
     - Point of No Return (PNR)
    """

    def __init__(self, split: Literal['train', 'val', 'validation', 'test'],
                 num_segments: int = 8,
                 root="data/ego4d",
                 features: Ego4dBackbones = "slowfast8x8_r101_k400",
                 version: int = 1,
                 transform=None, pre_transform=None, pre_filter=None,
                 aug_prob: float = 0.1,
                 remove_overlapping_segments: bool = False,
                 verbose=True):
        assert pre_transform is None, "This dataset does not support pre_transform"
        assert pre_filter is None, "This dataset does not support pre_filter"

        self.split = split.replace("validation", "val")
        self.verbose = verbose
        self.version = version
        self.features_path = features
        self.num_segments = num_segments
        self.stride = __features_strides__[features]
        self.aug_prob = aug_prob

        # Load OSCC annotations into a list of Ego4dOSCCEntry objects
        if not os.path.exists(osp.join(root, "raw", f"annotations/v{self.version}", f"fho_oscc-pnr_{self.split}.json")):
            raise FileNotFoundError(f"Could not find the OSCC annotations for split {self.split} in {osp.join(root, 'raw', 'annotations')}")

        self.annotations = json.load(open(osp.join(root, "raw", f"annotations/v{self.version}", f"fho_oscc-pnr_{self.split}.json"), 'r'))
        self.annotations = pd.DataFrame.from_records(self.annotations['clips'])

        avg_duration = (self.annotations.parent_end_sec - self.annotations.parent_start_sec).mean()
        logger.info(f"Average length of the OSCC segments in the {self.split} split: {avg_duration:.2f} seconds")

        if self.split == 'train' and remove_overlapping_segments:
            positives = self.annotations[~pd.isna(self.annotations.parent_pnr_frame)]
            negatives = self.annotations[pd.isna(self.annotations.parent_pnr_frame)]

            # merge positive and negative samples to check for overlapping segments
            joined = positives.merge(negatives, on='video_uid', suffixes=('_p', '_n'))

            joined['int_pos'] = joined.apply(lambda row: pd.Interval(row['parent_start_sec_p'], row['parent_end_sec_p'], closed='both'), axis=1)
            joined['int_neg'] = joined.apply(lambda row: pd.Interval(row['parent_start_sec_n'], row['parent_end_sec_n'], closed='both'), axis=1)

            # find overlapping intervals
            overlapping = joined.apply(lambda row: row['int_pos'].overlaps(row['int_neg']), axis=1)
            joined = joined[overlapping]

            overlapped_ids = pd.concat([joined.unique_id_p, joined.unique_id_n]).unique()
            logger.info(f"Found {len(overlapped_ids)} overlapping segments in the {self.split} split: removing them.")

            self.annotations = self.annotations[~self.annotations.unique_id.isin(overlapped_ids)]

        self.annotations = [
            Ego4dOSCCPNREntry(
                entry['video_uid'], entry['unique_id'],
                entry['parent_start_frame'], entry['parent_end_frame'],
                float(entry['parent_start_sec']), float(entry['parent_end_sec']),
                int(entry['state_change']) if 'state_change' in entry else -1,
                float(entry['parent_pnr_frame']) if 'state_change' in entry and not pd.isna(entry['parent_pnr_frame']) else None
            )
            for _, entry in self.annotations.iterrows()
        ]

        if 'egovlp' in features:
            remove_list = [
                '77ed1624-f87b-4196-9a0a-95b7023b18e4',
                'd18ef16d-f803-4387-bb5e-7876f1522a63',
                '8e914832-2dd1-44fd-81f8-1b7e2ccd2402'
            ]
            self.annotations = [entry for entry in self.annotations if entry.video_uid not in remove_list]

        # Load the list of unique video ids
        self.video_uids = list(set((entry.video_uid for entry in self.annotations)))

        # Initialize the dataset
        super().__init__(root, transform, pre_transform, pre_filter, verbose)

        if self.verbose:
            logger.info(f"Created dataset for Ego4D - OSCC for split {self.split}. Dataset contains {len(self)} samples.")

        # Load memory mapped features
        self._features = self._load_memmapped_features()

    @property
    def _raw_features_path(self):
        return osp.join(self.raw_dir, 'features', self.features_path)

    def _load_memmapped_features(self):
        # Example of path: data/ego4d/processed/omnivore_image_swinl/fho_train.csv
        features_path = osp.join(self.processed_dir, 'features', self.features_path)

        # sanity checks: verify that all the features were correctly converted to numpy arrays
        # for video_uid in self.video_uids:
        #     np_features = np.load(osp.join(features_path, f'{video_uid}.npy'), mmap_mode='r')
        #     torch_features = torch.load(osp.join(self._raw_features_path, f'{video_uid}.pt'))
        #     assert np.allclose(np_features, torch_features.numpy()), f"Something went wrong when loading features for video_uid {video_uid}"

        return {
            video_uid: np.load(osp.join(features_path, f'{video_uid}.npy'), mmap_mode='r')
            for video_uid in self.video_uids
        }

    @property
    def features_size(self) -> int:
        return __features_sizes__[self.features_path]

    @property
    def processed_file_names(self):
        filenames = [f'features/{self.features_path}/{video_uid}.npy' for video_uid in self.video_uids]
        filenames += [f'features/{self.features_path}/oscc_{self.split}_v{self.version}.csv']
        return filenames

    def process(self):
        features_path = osp.join(self.processed_dir, 'features', self.features_path)
        os.makedirs(features_path, exist_ok=True)

        metadata = []
        for video_uid in tqdm(self.video_uids):
            if not os.path.exists(osp.join(self._raw_features_path, f'{video_uid}.pt')):
                print(f"Could not find features for video {video_uid} in {self._raw_features_path}")
                continue

            features = torch.load(osp.join(self._raw_features_path, f'{video_uid}.pt'))
            metadata.append((video_uid, features.shape[0], features.shape[1]))

            # sf_feat = np.load(osp.join(features_path.replace('egovlpv2', 'slowfast8x8_r101_k400'), f'{video_uid}.npy'))
            # if features.shape[0] != sf_feat.shape[0]:
            #     print(f"Features for video {video_uid} have different number of frames")
            #     # os.remove(osp.join(self._raw_features_path, f'{video_uid}.pt'))

            if os.path.exists(osp.join(features_path, f'{video_uid}.npy')):
                continue

            # Resave the features as a numpy array
            np.save(osp.join(features_path, f'{video_uid}.npy'), features)

        # Save metadata
        metadata = pd.DataFrame(metadata, columns=['video_uid', 'length', 'features_size'])
        metadata.to_csv(osp.join(features_path, f'oscc_{self.split}_v{self.version}.csv'), index=False)

    def len(self) -> int:
        return len(self.annotations)

    def get(self, idx):
        segment = self.annotations[idx]
        state_change = segment.state_change
        video_features = self._features[segment.video_uid]

        start_frame = segment.start_frame - (segment.start_frame % self.stride)
        end_frame = segment.end_frame - (segment.end_frame % self.stride)

        n_segments = (end_frame - start_frame) // self.stride

        # pick (4 * 3) segments out of n_segments
        if self.split == 'train':
            selected_segments = np.random.choice(n_segments, size=4 * self.num_segments, replace=(n_segments < 4 * self.num_segments))
        else:
            selected_segments = np.linspace(0, n_segments, num=4 * self.num_segments, endpoint=False, dtype=int)
        selected_segments.sort()

        try:
            graph = np.take(video_features[start_frame//self.stride:end_frame//self.stride], selected_segments, axis=0)
        except:
            graph = np.zeros((len(selected_segments), video_features.shape[1]), dtype=np.float32)
        graph = rearrange(graph, '(n s) h -> n s h', n=4, s=self.num_segments, h=graph.shape[-1])

        if self.split == 'train' and state_change and random.random() < self.aug_prob:
            # randomly choose a frame to replace
            pnr_segment = max((i for i, (sel_segment) in enumerate(selected_segments) if (start_frame + sel_segment * self.stride) < segment.pnr_frame), default=0)
            if pnr_segment > 0:
                graph = graph[:pnr_segment] + [graph[pnr_segment - 1]] * (len(graph) - pnr_segment)
            else:
                graph = [graph[1], *graph[1:]]
            state_change = 0

        return Data(x=torch.from_numpy(graph), y=state_change, pos=torch.arange(0, len(graph)), uid=segment.unique_uid, video_uid=segment.video_uid)


class Ego4dPNRDataset(Ego4dOSCCDataset):

    def __init__(self, split: Literal['train', 'val', 'validation', 'test'],
                 num_segments: int = 8, root="data/ego4d",
                 features: Ego4dBackbones = "slowfast8x8_r101_k400",
                 version: int = 1,
                 transform=None, pre_transform=None, pre_filter=None, verbose=True):
        super().__init__(split, num_segments, root, features, version, transform, pre_transform, pre_filter, verbose)

        # remove annotations with a PNR frame (no state change)
        self.annotations = [entry for entry in self.annotations if entry.pnr_frame is not None or 'test' in self.split]

    def get(self, idx):
        segment: Ego4dOSCCPNREntry = self.annotations[idx]
        video_features = self._features[segment.video_uid]

        # relative position of the pnr frame wrt the start frame
        pnr_frame = segment.pnr_frame
        start_frame, end_frame = segment.start_frame, segment.end_frame

        if self.split == "train":
            random_length_seconds = np.random.uniform(5, 8)
            random_start_seconds = segment.start_sec + np.random.uniform(8 - random_length_seconds)
            start_frame = np.floor(random_start_seconds * 30).astype(np.int32)
            random_end_seconds = random_start_seconds + random_length_seconds
            if random_end_seconds > segment.end_sec:
                random_end_seconds = segment.end_sec
            end_frame = np.floor(random_end_seconds * 30).astype(np.int32)
            if segment.pnr_frame > end_frame:
                end_frame = segment.end_frame
            if segment.pnr_frame < start_frame:
                start_frame = segment.start_frame

        # randomly choose a set of frames to extract the features from
        candidate_frames = np.linspace(start_frame, end_frame, num=self.num_segments, dtype=int, endpoint=False)
        candidate_frames = np.clip(candidate_frames, start_frame, end_frame)

        # round below the indices
        frame_indices_low = np.clip(np.floor(candidate_frames / self.stride).astype(int), 0, video_features.shape[0] - 1)
        # round above the indices
        frame_indices_high = np.clip(np.ceil(candidate_frames / self.stride).astype(int), 0, video_features.shape[0] - 1)

        try:
            low_features = np.take(video_features, frame_indices_low, axis=0)
            high_features = np.take(video_features, frame_indices_high, axis=0)
        except:
            low_features = np.zeros((len(frame_indices_low), video_features.shape[1]), dtype=np.float32)
            high_features = np.zeros((len(frame_indices_high), video_features.shape[1]), dtype=np.float32)

        features = np.zeros_like(low_features, dtype=np.float32)
        # when low and high indices do not match, we need to interpolate the features
        features = (1 - (candidate_frames % self.stride) / self.stride)[:, np.newaxis] * low_features
        features += ((candidate_frames % self.stride) / self.stride)[:, np.newaxis] * high_features
        # when low and high indices match, we can just copy the features
        features[frame_indices_low == frame_indices_high] = low_features[frame_indices_low == frame_indices_high]

        # convert the absolute distance of the pnr frame into a onehot vector
        if 'test' not in self.split:
            distances = torch.from_numpy(np.abs(candidate_frames - pnr_frame)).long()
            labels = torch.zeros_like(distances).long()
            labels[distances.argmin()] = 1
        else:
            labels = -1 * torch.ones((len(candidate_frames), )).long()

        return Data(
            x=torch.from_numpy(features).float().unsqueeze(1).repeat(1, 3, 1),
            y=labels,
            video_uid=segment.video_uid,
            uid=segment.unique_uid,
            unique_uid=segment.unique_uid,
            pos=torch.arange(0, features.shape[0]),
            pnr_frame=segment.pnr_frame,
            start_frame=start_frame,
            end_frame=end_frame,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec
        )


if __name__ == "__main__":
    print("OSCC Train dataset:")
    dataset = Ego4dOSCCDataset("train", version=1)
    print(" - Number of samples: ", len(dataset))
    print(" - Number of unique video ids: ", len(dataset.video_uids))
    print(" - Number of memory mapped features: ", len(dataset._features))

    assert len(dataset) == 41_085, "Wrong number of samples"
    assert sum([entry.state_change for entry in dataset.annotations]) == 20_041, "Wrong number of positive samples"
    assert sum([1 - entry.state_change for entry in dataset.annotations]) == 21_044, "Wrong number of negative samples"

    print(" - Positive samples: ", sum([entry.state_change for entry in dataset.annotations]))
    print(" - Negative samples: ", len(dataset.annotations) - sum([entry.state_change for entry in dataset.annotations]))
    # print(dataset[42])  # positive sample
    # print(dataset[41_000])  # negative sample
    print()

    print("OSCC Validation dataset:")
    dataset = Ego4dOSCCDataset("val", version=1)
    print(" - Number of samples: ", len(dataset))
    print(" - Number of unique video ids: ", len(dataset.video_uids))
    print(" - Number of memory mapped features: ", len(dataset._features))

    assert len(dataset) == 28_348, "Wrong number of samples"
    assert sum([entry.state_change for entry in dataset.annotations]) == 13_628, "Wrong number of positive samples"
    assert sum([1 - entry.state_change for entry in dataset.annotations]) == 14_720, "Wrong number of negative samples"

    print(" - Positive samples: ", sum([entry.state_change for entry in dataset.annotations]))
    print(" - Negative samples: ", sum([1 - entry.state_change for entry in dataset.annotations]))
    print()

    print("PNR Train dataset:")
    dataset = Ego4dPNRDataset("train", version=1, num_segments=16)
    dataset[42]
    import pdb; pdb.set_trace()
    print(" - Number of samples: ", len(dataset))
    print(" - Number of unique video ids: ", len(dataset.video_uids))
    print()

    print("PNR Validation dataset:")
    dataset = Ego4dPNRDataset("val", version=1, num_segments=16)
    print(" - Number of samples: ", len(dataset))
    print(" - Number of unique video ids: ", len(dataset.video_uids))
