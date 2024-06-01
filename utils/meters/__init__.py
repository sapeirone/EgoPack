import torch

from data.base_dataset import BaseDataset
from data.ego4d_fho import Ego4dRecognitionDataset, Ego4dAnticipationDataset, Ego4dLTADataset
from data.ego4d_oscc import Ego4dOSCCDataset, Ego4dPNRDataset
from .base import BaseMeter
from .ego4d import Ego4dRecognitionMeter, Ego4dAnticipationMeter, Ego4dOSCCMeter, Ego4dLTAMeter, Ego4dPNRMeter


def build_meter_for_dataset(dataset: BaseDataset, save_features: bool = False, device: torch.device = "cuda") -> BaseMeter:
    if isinstance(dataset, (Ego4dRecognitionDataset, )):
        return Ego4dRecognitionMeter(dataset, save_features=save_features, device=device)
    elif isinstance(dataset, (Ego4dAnticipationDataset, )):
        return Ego4dAnticipationMeter(dataset, device=device)
    elif isinstance(dataset, (Ego4dPNRDataset, )):
        return Ego4dPNRMeter(dataset, device=device)
    elif isinstance(dataset, (Ego4dOSCCDataset, )):
        return Ego4dOSCCMeter(dataset, device=device)
    elif isinstance(dataset, (Ego4dLTADataset, )):
        return Ego4dLTAMeter(dataset, device=device)
    else:
        raise NotImplementedError
