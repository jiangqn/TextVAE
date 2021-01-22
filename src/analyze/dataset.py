import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from typing import List, Tuple

class RegressionDataset(Dataset):

    def __init__(self, sample_path: str, latent_variable_path: str, target: str) -> None:

        sample_data = pd.read_csv(sample_path, delimiter="\t")
        latent_variable = np.load(latent_variable_path)

        assert len(latent_variable.shape) == 2
        assert len(sample_data) == latent_variable.shape[0]

        self.latent_variable = torch.from_numpy(latent_variable).float()
        self.target = torch.from_numpy(np.asarray(sample_data[target])).float()

    def __len__(self) -> int:
        return self.target.size(0)

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.latent_variable[item], self.target[item]

class ClassificationDataset(Dataset):

    def __init__(self, sample_path: str, latent_variable_path: str, target: str) -> None:

        sample_data = pd.read_csv(sample_path, delimiter="\t")
        latent_variable = np.load(latent_variable_path)

        assert len(latent_variable.shape) == 2
        assert len(sample_data) == latent_variable.shape[0]

        self.latent_variable = torch.from_numpy(latent_variable).float()
        self.target = torch.from_numpy(np.asarray(sample_data[target])).long()

    def __len__(self) -> int:
        return self.target.size(0)

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.latent_variable[item], self.target[item]