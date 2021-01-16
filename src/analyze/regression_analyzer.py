import torch
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from src.analyze.iresnet import InvertibleResNet
from src.analyze.dataset import RegressionDataset
from src.analyze.multiple_correlation import multiple_correlation, get_linear_weights
import os
from typing import Tuple
from copy import deepcopy

class RegressionAnalyzer(object):

    def __init__(self, base_path: str, target_name: str, hidden_size: int, n_blocks: int) -> None:
        super(RegressionAnalyzer, self).__init__()
        self.base_path = base_path
        self.target_name = target_name
        self.model = InvertibleResNet(hidden_size=hidden_size, n_blocks=n_blocks, output_size=1)
        self.model = self.model.cuda()

    def fit(self, batch_size: int = 100, lr: float = 3e-4, momentum: float = 0.8, weight_decay: float = 3e-4, num_epoches: int = 10) -> None:
        train_dataset = RegressionDataset(
            sample_path=os.path.join(self.base_path, "vanilla_sample_train.tsv"),
            latent_variable_path=os.path.join(self.base_path, "vanilla_sample_train.npy"),
            targets=[self.target_name]
        )
        dev_dataset = RegressionDataset(
            sample_path=os.path.join(self.base_path, "vanilla_sample_dev.tsv"),
            latent_variable_path=os.path.join(self.base_path, "vanilla_sample_dev.npy"),
            targets=[self.target_name]
        )
        test_dataset = RegressionDataset(
            sample_path=os.path.join(self.base_path, "vanilla_sample_test.tsv"),
            latent_variable_path=os.path.join(self.base_path, "vanilla_sample_test.npy"),
            targets=[self.target_name]
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        dev_loader = DataLoader(
            dataset=dev_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )
        self.criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        min_dev_loss = None
        best_model = None
        for epoch in range(num_epoches):

            total_samples = 0
            total_loss = 0

            for i, data in enumerate(train_loader):

                self.model.train()
                optimizer.zero_grad()

                latent_variable, target = data
                latent_variable, target = latent_variable.cuda(), target.cuda()

                prediction = self.model(latent_variable)
                loss = self.criterion(prediction, target)
                loss.backward()
                optimizer.step()

                batch_size = target.size(0)
                total_samples += batch_size
                total_loss += loss.item() * batch_size

                if i % 100 == 0:
                    train_loss = total_loss / total_samples
                    total_samples = 0
                    total_loss = 0
                    dev_loss, dev_correlation, dev_transformed_correlation = self.eval(dev_loader)
                    print("[epoch %4d] [step %4d] [train_loss: %.4f] [dev_loss: %.4f\tdev_correlation: %.4f\tdev_transformed_correlation: %.4f]" %
                          (epoch, i, train_loss, dev_loss, dev_correlation, dev_transformed_correlation))
                    if min_dev_loss == None or dev_loss < min_dev_loss:
                        min_dev_loss = dev_loss
                        best_model = deepcopy(self.model)

        self.model = best_model
        _, test_correlation, test_transformed_correlation = self.eval(test_loader)
        print("test_correlation: %.4f\ttest_transformed_correlation: %.4f" % (test_correlation, test_transformed_correlation))

        self.save_linear_weights()

    def eval(self, data_loader: DataLoader) -> Tuple[float, float, float]:

        self.model.eval()

        latent_variable_list = []
        transformed_latent_variable_list = []
        target_list = []

        total_samples = 0
        total_loss = 0

        with torch.no_grad():

            for data in data_loader:

                latent_variable, target = data
                latent_variable, target = latent_variable.cuda(), target.cuda()

                prediction = self.model(latent_variable)
                loss = self.criterion(prediction, target)
                batch_size = target.size(0)
                total_samples += batch_size
                total_loss += batch_size * loss.item()

                transformed_latent_variable = self.model.transform(latent_variable)
                latent_variable_list.append(latent_variable)
                transformed_latent_variable_list.append(transformed_latent_variable)
                target_list.append(target)

        loss = total_loss / total_samples

        latent_variable = torch.cat(latent_variable_list, dim=0)
        transformed_latent_variable = torch.cat(transformed_latent_variable_list, dim=0)
        target = torch.cat(target_list, dim=0)
        target = target.squeeze(-1)

        correlation = multiple_correlation(latent_variable, target)
        transformed_correlation = multiple_correlation(transformed_latent_variable, target)

        return loss, correlation, transformed_correlation

    def get_data(self, batch_size: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        test_dataset = RegressionDataset(
            sample_path=os.path.join(self.base_path, "vanilla_sample_test.tsv"),
            latent_variable_path=os.path.join(self.base_path, "vanilla_sample_test.npy"),
            targets=[self.target_name]
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        self.model.eval()

        latent_variable_list = []
        transformed_latent_variable_list = []
        target_list = []

        with torch.no_grad():
            for data in test_loader:
                latent_variable, target = data
                latent_variable, target = latent_variable.cuda(), target.cuda()

                transformed_latent_variable = self.model.transform(latent_variable)
                latent_variable_list.append(latent_variable)
                transformed_latent_variable_list.append(transformed_latent_variable)
                target_list.append(target)

        latent_variable = torch.cat(latent_variable_list, dim=0)
        transformed_latent_variable = torch.cat(transformed_latent_variable_list, dim=0)
        target = torch.cat(target_list, dim=0)
        target = target.squeeze(-1)

        return latent_variable, transformed_latent_variable, target

    def save_linear_weights(self) -> None:

        latent_variable, transformed_latent_variable, target = self.get_data()

        latent_linear_weights = get_linear_weights(latent_variable, target)
        latent_weight = latent_linear_weights[0:-1]
        latent_weight = latent_weight / latent_weight.norm()
        self.latent_weight = latent_weight
        self.latent_projection_dict = self.get_projection_dict(latent_variable, latent_weight, target)

        transformed_latent_linear_weights = get_linear_weights(transformed_latent_variable, target)
        transformed_latent_weight = transformed_latent_linear_weights[0:-1]
        transformed_latent_weight = transformed_latent_weight / transformed_latent_weight.norm()
        self.transformed_latent_weight = transformed_latent_weight
        self.transformed_latent_projection_dict = self.get_projection_dict(transformed_latent_variable, transformed_latent_weight, target)

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

    def transformed_latent_variable_transform(self, latent_variable: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        """
        :param latent_variable: torch.FloatTensor (num, latent_size)
        :param target: torch.LongTensor (num,)
        :return : torch.FloatTensor (num, latent_size)
        """

        weight = self.transformed_latent_weight.unsqueeze(-1)
        current_projection = latent_variable.matmul(weight)
        target_projection = self.latent_projection_dict[target].unsqueeze(-1)
        latent_variable = latent_variable + (target_projection - current_projection).matmul(weight.transpose(0, 1))
        return latent_variable