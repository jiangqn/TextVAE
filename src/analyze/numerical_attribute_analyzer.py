import torch
from src.analyze.dataset import RegressionDataset
from torch.utils.data.dataloader import DataLoader
from src.analyze.multiple_correlation import multiple_correlation, get_linear_weights
import os
from typing import Tuple

class NumericalAttributeAnalyzer(object):

    def __init__(self, base_path: str, target: str, latent_size: int) -> None:
        super(NumericalAttributeAnalyzer, self).__init__()
        self.base_path = base_path
        self.target = target
        self.latent_size = latent_size

    def fit(self, batch_size: int = 100) -> None:
        latent_variable, target = self.get_data(batch_size, "train")
        weight = get_linear_weights(latent_variable, target)[0:-1]
        weight = weight / weight.norm()
        self.latent_weight = weight
        latent_variable, target = self.get_data(batch_size, "test")
        print("multiple correlation: %.4f" % multiple_correlation(latent_variable, target))
        self.latent_projection_dict = self.get_projection_dict(latent_variable, weight, target)

    def get_data(self, batch_size: int = 100, division: str = "test") -> Tuple[torch.Tensor, torch.Tensor]:

        sample_path = os.path.join(self.base_path, "vanilla_sample_%s.tsv" % division)
        latent_variable_path = os.path.join(self.base_path, "vanilla_sample_%s.npy" % division)

        dataset = RegressionDataset(
            sample_path=sample_path,
            latent_variable_path=latent_variable_path,
            target=self.target
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        latent_variable_list = []
        target_list = []

        with torch.no_grad():
            for data in data_loader:
                latent_variable, target = data
                latent_variable, target = latent_variable.cuda(), target.cuda()

                latent_variable_list.append(latent_variable)
                target_list.append(target)

        latent_variable = torch.cat(latent_variable_list, dim=0)
        target = torch.cat(target_list, dim=0)

        return latent_variable, target

    def get_projection_dict(self, latent_variable: torch.Tensor, weight: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        """
        :param latent_variable: torch.FloatTensor (num, latent_size)
        :param weight: torch.FloatTensor (latent_size,)
        :param target: torch.FloatTensor (num,)
        :return projection_dict: torch.FloatTensor (max_value + 1,)
        """

        weight = weight / weight.norm()
        projection = latent_variable.matmul(weight)

        min_value = int(target.min().item() + 1e-6)
        max_value = int(target.max().item() + 1e-6)
        projection_dict = torch.zeros(max_value + 1, dtype=torch.float, device=latent_variable.device)
        for i in range(min_value, max_value + 1):
            projection_dict[i] = projection[target == i].mean()

        return projection_dict

    def latent_variable_transform(self, latent_variable: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        """
        :param latent_variable: torch.FloatTensor (num, latent_size)
        :param target: torch.LongTensor (num,)
        :return : torch.FloatTensor (num, latent_size)
        """

        weight = self.latent_weight.unsqueeze(-1)
        current_projection = latent_variable.matmul(weight)
        target_projection = self.latent_projection_dict[target].unsqueeze(-1)
        latent_variable = latent_variable + (target_projection - current_projection).matmul(weight.transpose(0, 1))
        return latent_variable