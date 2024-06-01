import torch

from typing import Tuple, Optional

import numpy as np


class BaseDataset(torch.utils.data.Dataset): # type: ignore
    """Base class for all video datasets.

    This class supports annotations with multiple separate class labels and
    possibly a joint label that combines the separate labels.
    Example: samples from EK100 are annotated with a verb and a noun labels.
    Additionally, samples are also annotated with an action label which is a
    combination of verb and noun: action = (verb, noun).

    TODO: improve documentation on joint classes and masks.
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, verbose=True):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.verbose = verbose

    @property
    def num_labels(self) -> int:
        """Number of labels associated to each sample, including joint labels.

        Returns
        -------
        int
            the number of labels associated to each sample
        """
        raise NotImplementedError()

    @property
    def label_names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    @property
    def has_joint_label(self) -> bool:
        """Whether the samples have a joint label.

        Returns
        -------
        int
            True if the samples of this dataset have a joint label
        """
        raise NotImplementedError()

    @property
    def joint_label_masks(self) -> Optional[Tuple[torch.Tensor, ...]]:
        """Return the masks for the individual labels, given the joint label.

        If the dataset has no joint label, this property should be set to None.
        Otherwise, the size of the returned tuple must match self.num_labels - 1.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            masks for the joint label
        """
        raise NotImplementedError()

    @property
    def tail_classes(self) -> Tuple[torch.Tensor, ...]:
        """Tail classes of this dataset.

        The definition of tail classes varies depending on the dataset.
        Some datasets may not have tail classes.

        If the samples are annotated with multiple separate classes and
        a joint class, e.g. EK100 has verb, noun and action labels, this method should
        return first the separate classes followed by the joint class.

        The size of the returned tuple must be self.num_labels.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            tail classes of the dataset
        """
        raise NotImplementedError()

    @property
    def class_labels(self) -> Tuple[str, ...]:
        """Class labels of this dataset.

        If the samples are annotated with multiple separate classes and
        a joint class, e.g. EK100 has verb, noun and action labels, this method should
        return first the separate classes followed by the joint class.

        The size of the returned tuple must be self.num_labels.

        Returns
        -------
        Tuple[str, ...]
            class labels of the dataset
        """
        raise NotImplementedError()

    @property
    def num_class_labels(self) -> Tuple[int, ...]:
        """Return the nunmber of class labels.

        The size of the returned tuple must be self.num_labels.

        Returns
        -------
        Tuple[int, ...]
            number of class labels
        """
        raise NotImplementedError()

    @property
    def features_size(self) -> int:
        """Features size.

        Returns
        -------
        int
            features_size
        """
        raise NotImplementedError()


class BaseFrameDataset(BaseDataset):

    @classmethod
    def random_sampling_indices(cls, size, n):
        average_duration = size // n
        if average_duration > 0:
            indices = np.multiply(list(range(n)), size / n)
            indices = indices + np.random.randint(average_duration, size=n)
            indices = np.clip(indices, 0, size)
        else:
            indices = np.linspace(0, size, n, endpoint=False, dtype=int)

        indices = np.round(indices)
        return indices.astype(int)

    @classmethod
    def uniform_sampling_indices(cls, size, n):
        offsets = np.linspace(0, size, n, endpoint=False, dtype=int)
        offsets = offsets + (size // n // 2)
        return offsets.astype(int)

    @classmethod
    def random_sampling(cls, data: np.ndarray, num_segments: int):
        indices = BaseFrameDataset.random_sampling_indices(data.shape[0], num_segments)
        return np.take(data, indices, axis=0)

    @classmethod
    def uniform_sampling(cls, data: np.ndarray, num_segments: int):
        indices = BaseFrameDataset.uniform_sampling_indices(data.shape[0], num_segments)
        return np.take(data, indices, axis=0)


class BaseVideoDataset(BaseDataset):
    pass


# extract the feature for a particular key
def extract_by_key(envs, key):
    '''
        input:
            env: lmdb environment loaded (see main function)
            key: the frame number in lmdb key format for which the feature is to be extracted
                 the lmdb key format is '{sequence_name}/{view_name}/{view_name}_{frame_no:010d}.jpg'
                 e.g. nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/C10095_rgb/C10095_rgb_0000000001.jpg
        output: a 2048-D np-array (TSM feature corresponding to the key)
    '''
    for env in envs:
        with env.begin() as e:
            data = e.get(key.strip().encode('utf-8'))
            if data is not None:
                return torch.tensor(np.frombuffer(data, 'float32'))
    return None


if __name__ == '__main__':
    print("Random:")
    print(BaseFrameDataset.random_sampling_indices(5, 8))
    print(BaseFrameDataset.random_sampling_indices(10, 8))
    print(BaseFrameDataset.random_sampling_indices(100, 8))
    print(BaseFrameDataset.random_sampling_indices(100, 8))
    print(BaseFrameDataset.random_sampling_indices(100, 8))

    print()

    print("Uniform:")
    print(BaseFrameDataset.uniform_sampling_indices(10, 8))
    print(BaseFrameDataset.uniform_sampling_indices(100, 8))
