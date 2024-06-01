import json
import os
import os.path as osp
import logging
from typing import Literal, Tuple, List, Optional
from collections import namedtuple

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


Ego4dFHOEntry = namedtuple("Ego4dFHOEntry", ["id", "video_uid", "clip_uid", "start_frame", "end_frame", "verb_label", "noun_label"])

Ego4dAREntry = namedtuple("Ego4dAREntry", ["video_uid", "clip_uid", "actions"])
Ego4dLTAEntry = namedtuple("Ego4dLTAEntry", ["video_uid", "clip_uid", "id", "input_clips", "forecast_clips"])


class Ego4dFHODataset(BaseFrameDataset, Dataset):
    """Ego4D datase for the First-Person Hand Object Interaction (FHO) task(s).

    This dataset can support the following tasks:
     - Action Recognition (AR)
     - Action Anticipation (AA)
    """

    def __init__(self, split: Literal['train', 'val', 'validation', 'test'],
                 root="data/ego4d",
                 features: Ego4dBackbones = "slowfast8x8_r101_k400",
                 version: int = 1,
                 transform=None, pre_transform=None, pre_filter=None,
                 verbose=True):
        assert pre_transform is None, "This dataset does not support pre_transform"
        assert pre_filter is None, "This dataset does not support pre_filter"

        self.split = split.replace("validation", "val")
        self.verbose = verbose
        self.version = version
        self.features_path = features
        self.stride = __features_strides__[features]

        # Load FHO annotations into a list of Ego4dFHOEntry objects
        if not os.path.exists(osp.join(root, "raw", f"annotations/v{version}", f"fho_lta_{self.split}.json")):
            raise FileNotFoundError(f"Could not find the FHO annotations for split {self.split} in {osp.join(root, 'raw', 'annotations')}")

        self.annotations = json.load(open(osp.join(root, "raw", f"annotations/v{version}", f"fho_lta_{self.split}.json"), 'r'))
        self.annotations = [
            Ego4dFHOEntry(entry['action_idx'], entry['video_uid'], entry['clip_uid'],
                          entry['clip_parent_start_frame'] + entry['action_clip_start_frame'],
                          entry['clip_parent_start_frame'] + entry['action_clip_end_frame'],
                          entry['verb_label'] if 'verb_label' in entry else None, entry['noun_label'] if 'noun_label' in entry else None)
            for entry in self.annotations['clips']
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
        self.clip_uids = list(set((entry.clip_uid for entry in self.annotations)))

        # Initialize the dataset
        super().__init__(root, transform, pre_transform, pre_filter, verbose)

        # Read taxonomy
        if not os.path.exists(osp.join(self.raw_dir, f"annotations/v{version}", "fho_lta_taxonomy.json")):
            raise FileNotFoundError(f"Could not find the FHO taxonomy in {osp.join(root, 'raw')}")

        self.taxonomy = json.load(open(osp.join(self.raw_dir, f"annotations/v{version}", "fho_lta_taxonomy.json"), "r"))

        # Load memory mapped features
        self._features = self._load_memmapped_features()

    @property
    def _raw_features_path(self):
        return osp.join(self.raw_dir, 'features', self.features_path)

    def _load_memmapped_features(self):
        # Example of path: data/ego4d/processed/omnivore_image_swinl/fho_train.csv
        features_path = osp.join(self.processed_dir, 'features', self.features_path)
        logger.info(f"Loading features from {features_path}...")

        return {
            video_uid: np.load(osp.join(features_path, f'{video_uid}.npy'), mmap_mode='r')
            for video_uid in self.video_uids
        }

    @property
    def label_names(self) -> Tuple[str, ...]:
        return tuple(['verbs', 'nouns'])

    @property
    def num_labels(self) -> int:
        return len(self.label_names)

    @property
    def has_joint_label(self) -> bool:
        return False

    @property
    def tail_classes(self):
        return ([],)

    @property
    def class_labels(self):
        return tuple([self.taxonomy[label] for label in self.label_names])

    @property
    def num_class_labels(self) -> Tuple[int, ...]:
        # return tuple(len(labels) for labels in self.class_labels)
        return tuple(len(labels) for labels in self.class_labels)

    @property
    def features_size(self) -> int:
        return __features_sizes__[self.features_path]

    @property
    def processed_file_names(self):
        filenames = [f'features/{self.features_path}/{video_uid}.npy' for video_uid in self.video_uids]
        filenames += [f'features/{self.features_path}/fho_{self.split}_v{self.version}.csv']
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
            # sf_feat = np.load(osp.join(features_path.replace('egovlpv2', 'slowfast8x8_r101_k400'), f'{video_uid}.npy'))

            # if features.shape[0] != sf_feat.shape[0]:
            #     print(f"Features for video {video_uid} have different number of frames")
            #     os.remove(osp.join(self._raw_features_path, f'{video_uid}.pt'))
            metadata.append((video_uid, features.shape[0], features.shape[1]))

            if os.path.exists(osp.join(features_path, f'{video_uid}.npy')):
                continue

            # Resave the features as a numpy array
            np.save(osp.join(features_path, f'{video_uid}.npy'), features)

        # Save metadata
        metadata = pd.DataFrame(metadata, columns=['video_uid', 'length', 'features_size'])
        metadata.to_csv(osp.join(features_path, f'fho_{self.split}_v{self.version}.csv'), index=False)

    def len(self) -> int:
        return len(self.video_uids)

    def get(self, idx):
        raise NotImplementedError()


class Ego4dRecognitionDataset(Ego4dFHODataset):

    def __init__(self, split: Literal['train', 'val', 'test'],
                 num_segments: int = 8,
                 root="data/ego4d",
                 features: Ego4dBackbones = "slowfast8x8_r101_k400",
                 version: int = 1,
                 transform=None, pre_transform=None, pre_filter=None,
                 window_size: int = 9,
                 randomize_train: bool = True,
                 verbose=True):
        self.num_segments = num_segments
        self.randomize_train = randomize_train
        super().__init__(split, root, features, version, transform, pre_transform, pre_filter, verbose)

        # Group annotations by video id
        self.clip_annotations = {
            clip_uid: sorted([entry for entry in self.annotations if entry.clip_uid == clip_uid], key=lambda x: x.id)
            for clip_uid in self.clip_uids
        }

        self.window_size = window_size
        self.action_segments = []
        for clip_uid, actions in self.clip_annotations.items():
            video_uid = actions[0].video_uid

            for i in range(len(actions)):
                selected_actions = []

                left, right = i - (window_size // 2), i + (window_size - window_size // 2)

                selected_actions += [0] * max(0, -left)
                selected_actions += np.arange(max(0, left), min(len(actions), right)).tolist()

                selected_actions += [len(actions) - 1] * max(0, right - len(actions))
                self.action_segments.append(Ego4dAREntry(video_uid, clip_uid, [actions[sel_idx] for sel_idx in selected_actions]))

    def len(self) -> int:
        return len(self.action_segments)

    def get(self, idx):
        action_segment = self.action_segments[idx]

        video_uid = action_segment.video_uid
        
        verb_labels = torch.LongTensor([action.verb_label if (i == self.window_size // 2) else -1 for i, action in enumerate(action_segment.actions)])
        noun_labels = torch.LongTensor([action.noun_label if (i == self.window_size // 2) else -1 for i, action in enumerate(action_segment.actions)])
        pos = torch.arange(0, len(action_segment.actions)).long() - (self.window_size // 2)

        video_features = self._features[video_uid]

        graph: List[np.ndarray] = []
        for action in action_segment.actions:
            action_start = action.start_frame // self.stride
            action_end = min(video_features.shape[0] - 1, action.end_frame // self.stride)

            try:
                if self.split == "train" and self.randomize_train:
                    graph.append(BaseFrameDataset.random_sampling(video_features[action_start:action_end], self.num_segments))
                else:
                    graph.append(BaseFrameDataset.uniform_sampling(video_features[action_start:action_end], self.num_segments))
            except:
                graph.append(np.zeros((self.num_segments, self.features_size), dtype=np.float32))

        labels = torch.stack([verb_labels, noun_labels], dim=1)
        return Data(x=torch.from_numpy(np.stack(graph)), y=labels, pos=pos)


class Ego4dAnticipationDataset(Ego4dFHODataset):

    def __init__(self,
                 split: Literal['train', 'val', 'test'],
                 num_segments: int = 8,
                 root="data/ego4d",
                 features: Ego4dBackbones = "slowfast8x8_r101_k400",
                 anticipation_secs: int = 7,
                 blackout_secs: int = 1,
                 append_node: Optional[Literal['random', 'zero', 'avg']] = None,
                 version: int = 1,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 verbose=True):
        super().__init__(split, root, features, version, transform, pre_transform, pre_filter, verbose)
        self.num_segments = num_segments
        self.anticipation_secs = anticipation_secs
        self.blackout_secs = blackout_secs
        self.append_node = append_node

    def len(self) -> int:
        return len(self.annotations)

    def get(self, idx):
        action = self.annotations[idx]
        video_features = self._features[action.video_uid]

        data = []
        verb_labels, noun_labels = [], []

        for sec in range(-self.anticipation_secs, -self.blackout_secs):
            start = max(1, (action.start_frame + sec * 30) // self.stride) - 1
            end = max(1, (action.start_frame + (1 + sec) * 30) // self.stride)

            try:
                if self.split == "train":
                    data.append(BaseFrameDataset.random_sampling(video_features[start:end], self.num_segments))
                else:
                    data.append(BaseFrameDataset.uniform_sampling(video_features[start:end], self.num_segments))
            except:
                data.append(np.zeros((self.num_segments, self.features_size), dtype=np.float32))

            verb_labels.append(-1)
            noun_labels.append(-1)

        assert len(data) == len(verb_labels), "Mismatch between data and labels"

        if self.append_node is not None:
            if self.append_node == "random":
                data.append(np.random.random(data[-1].shape).astype(np.float32))
            elif self.append_node == "zero":
                data.append(np.zeros(data[-1].shape, dtype=np.float32))
            elif self.append_node == "avg":
                data.append(np.stack(data).mean(0))
            verb_labels.append(action.verb_label)
            noun_labels.append(action.noun_label)
        else:
            verb_labels[-1] = action.verb_label
            noun_labels[-1] = action.noun_label

        labels = torch.stack([torch.LongTensor(verb_labels), torch.LongTensor(noun_labels)], dim=1)
        pos = torch.arange(0, len(labels))
        return Data(x=torch.from_numpy(np.stack(data)), y=labels, pos=pos)


class Ego4dLTADataset(Ego4dFHODataset):

    def __init__(self, split: Literal['train', 'val', 'test_unannotated'],
                 num_segments: int = 8,
                 num_input_clips: int = 2,
                 num_forecasted_clips: int = 20,
                 append_node: Literal['random', 'zero', 'avg'] = 'avg',
                 root="data/ego4d",
                 features: Ego4dBackbones = "omnivore_video_swinl",
                 version: int = 1,
                 transform=None, pre_transform=None, pre_filter=None,
                 verbose=True):
        self.num_segments = num_segments
        super().__init__(split, root, features, version, transform, pre_transform, pre_filter, verbose)

        # lta attributes
        self.n_input_clips = num_input_clips
        self.n_forecast_clips = num_forecasted_clips
        self.append_node = append_node

        # Group annotations by video id
        self.clip_annotations = {
            clip_uid: sorted([entry for entry in self.annotations if entry.clip_uid == clip_uid], key=lambda x: x.id)
            for clip_uid in self.clip_uids
        }

        self.lta_annotations: List[Ego4dLTAEntry] = []
        for clip_uid, videos in self.clip_annotations.items():
            video_uid = videos[0].video_uid
            if 'test' in split:
                for i in range(len(videos) - num_input_clips):
                    input_clips = videos[i:i+num_input_clips]
                    self.lta_annotations.append(Ego4dLTAEntry(video_uid, clip_uid, videos[i+num_input_clips-1].id, input_clips, []))
            else:
                for i in range(len(videos) - num_input_clips - num_forecasted_clips):
                    input_clips = videos[i:i+num_input_clips]
                    forecast_clips = videos[i+num_input_clips:i+num_input_clips+num_forecasted_clips]
                    self.lta_annotations.append(Ego4dLTAEntry(video_uid, clip_uid, videos[i+num_input_clips-1].id, input_clips, forecast_clips))

    def len(self) -> int:
        return len(self.lta_annotations)

    def get(self, idx):
        lta_annotations = self.lta_annotations[idx]
        video_uid = lta_annotations.video_uid

        if 'test' in self.split:
            verb_labels = torch.LongTensor([-1] * len(lta_annotations.input_clips) + [0 for _ in range(self.n_forecast_clips)])
            noun_labels = torch.LongTensor([-1] * len(lta_annotations.input_clips) + [0 for _ in range(self.n_forecast_clips)])
        else:
            verb_labels = torch.LongTensor([-1] * len(lta_annotations.input_clips) + [clip.verb_label for clip in lta_annotations.forecast_clips])
            noun_labels = torch.LongTensor([-1] * len(lta_annotations.input_clips) + [clip.noun_label for clip in lta_annotations.forecast_clips])

        pos = torch.cat([
            torch.arange(0, self.n_input_clips).long(),
            torch.arange(self.n_input_clips, self.n_forecast_clips + self.n_input_clips).long()
        ])

        video_features = self._features[video_uid]

        input_clips: List[np.ndarray] = []
        for action in lta_annotations.input_clips:
            action_start = max(1, action.start_frame // self.stride) - 1
            action_end = min(video_features.shape[0] - 1, action.end_frame // self.stride)

            try:
                if self.split == "train":
                    input_clips.append(BaseFrameDataset.random_sampling(video_features[action_start:action_end], self.num_segments))
                else:
                    input_clips.append(BaseFrameDataset.uniform_sampling(video_features[action_start:action_end], self.num_segments))
            except:
                input_clips.append(np.zeros((self.num_segments, self.features_size), dtype=np.float32))

        forecast_clips: List[np.ndarray] = []
        for _ in range(self.n_forecast_clips):
            if self.append_node == "random":
                forecast_clips.append(np.random.random(input_clips[-1].shape).astype(np.float32))
            elif self.append_node == "zero":
                forecast_clips.append(np.zeros(input_clips[-1].shape, dtype=np.float32))
            else:
                forecast_clips.append(np.stack(input_clips).mean(0))

        graph = input_clips + forecast_clips

        labels = torch.stack([verb_labels, noun_labels], dim=1)
        return Data(x=torch.from_numpy(np.stack(graph)), y=labels, pos=pos, clip_uid=lta_annotations.clip_uid, last_idx=lta_annotations.id)


if __name__ == "__main__":
    dataset = Ego4dRecognitionDataset("train")
    print(f"Length of the recognition train dataset: {len(dataset)}")

    dataset = Ego4dRecognitionDataset("val")
    print(f"Length of the recognition val dataset: {len(dataset)}")

    dataset = Ego4dAnticipationDataset("train")
    print(f"Length of the anticipation train dataset: {len(dataset)}")

    dataset = Ego4dAnticipationDataset("val")
    print(f"Length of the anticipation val dataset: {len(dataset)}")

    dataset = Ego4dLTADataset("train")
    print(f"Length of the LTA train dataset: {len(dataset)}")

    dataset = Ego4dLTADataset("val")
    print(f"Length of the LTA val dataset: {len(dataset)}")
