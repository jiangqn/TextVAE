import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from typing import List

class RegressionDataset(Dataset):

    def __init__(self, sample_path: str, latent_variable_path: str, targets: List[str]) -> None:

        sample_data = pd.read_csv(sample_path, delimiter="\t")
        latent_variable = np.load(latent_variable_path)

        assert len(latent_variable.shape) == 2
        assert len(sample_data) == latent_variable.shape[0]

        self.latent_variable = torch.from_numpy(latent_variable).float()
        target = []
        for target_name in targets:
            target.append(np.asarray(sample_data[target_name]))
        target = np.stack(target, axis=1)
        self.target = torch.from_numpy(target).float()

    def __len__(self):
        return self.target.size(0)

    def __getitem__(self, item):
        return self.latent_variable[item], self.target[item]