import torch
from torch.utils.data.dataset import Dataset
import numpy as np

class RegressionDataset(Dataset):

    def __init__(self, feature: np.ndarray, target: np.ndarray) -> None:
        assert len(feature.shape) == 2
        assert len(target.shape) == 2
        assert feature.shape[0] == target.shape[0]
        self.feature = torch.from_numpy(feature).double()
        self.target = torch.from_numpy(target).double()

    def __len__(self):
        return self.target.size(0)

    def __getitem__(self, item):
        return self.feature[item], self.target[item]