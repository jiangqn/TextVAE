import torch
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from src.analyze.dataset import ClassificationDataset
from src.analyze.logistic_regression import LogisticRegression
import os
from typing import Tuple, List
from src.utils.tsv_max_entry_value import tsv_max_entry_value
from copy import deepcopy

class CategoricalAttributeAnalyzer(object):

    def __init__(self, base_path: str, latent_size: int) -> None:
        super(CategoricalAttributeAnalyzer, self).__init__()
        self.base_path = base_path
        self.latent_size = latent_size
        self.num_categories = tsv_max_entry_value(os.path.join(base_path, "train.tsv"), "label") + 1

    def fit(self, batch_size: int = 100, lr: float = 3e-4, num_epoches: int = 10) -> None:
        self.batch_size = batch_size
        train_dataset = ClassificationDataset(
            sample_path=os.path.join(self.base_path, "vanilla_sample_train.tsv"),
            latent_variable_path=os.path.join(self.base_path, "vanilla_sample_train.npy"),
            target="label"
        )
        dev_dataset = ClassificationDataset(
            sample_path=os.path.join(self.base_path, "vanilla_sample_dev.tsv"),
            latent_variable_path=os.path.join(self.base_path, "vanilla_sample_dev.npy"),
            target="label"
        )
        test_dataset = ClassificationDataset(
            sample_path=os.path.join(self.base_path, "vanilla_sample_test.tsv"),
            latent_variable_path=os.path.join(self.base_path, "vanilla_sample_test.npy"),
            target="label"
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
        self.model = LogisticRegression(latent_size=self.latent_size, output_size=self.num_categories)
        self.model = self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_model = None
        min_dev_loss = None
        for epoch in range(num_epoches):

            total_samples = 0
            correct_samples = 0
            total_loss = 0

            for i, data in enumerate(train_loader):

                self.model.train()
                optimizer.zero_grad()

                latent_variable, target = data
                latent_variable, target = latent_variable.cuda(), target.cuda()

                logit = self.model(latent_variable)
                loss = self.criterion(logit, target)
                loss.backward()
                optimizer.step()

                prediction = logit.argmax(dim=-1)
                batch_size = target.size(0)
                total_samples += batch_size
                correct_samples += (prediction == target).long().sum().item()
                total_loss += loss.item() * batch_size

                if i % 100 == 0:
                    train_loss = total_loss / total_samples
                    train_accuracy = correct_samples / total_samples
                    total_samples = 0
                    correct_samples = 0
                    total_loss = 0
                    dev_loss, dev_accuracy = self.eval(dev_loader)
                    print("[epoch %4d] [step %4d] [train_loss: %.4f\ttrain_accuracy: %.4f] [dev_loss: %.4f\tdev_accuracy: %.4f]" %
                          (epoch, i, train_loss, train_accuracy, dev_loss, dev_accuracy))
                    if min_dev_loss == None or dev_loss < min_dev_loss:
                        min_dev_loss = dev_loss
                        best_model = self.model

        self.model = deepcopy(best_model)
        test_loss, test_accuracy = self.eval(test_loader)
        print("test_accuracy: %.4f" % test_accuracy)

    def eval(self, data_loader: DataLoader) -> Tuple[float, float]:

        self.model.eval()

        total_samples = 0
        correct_samples = 0
        total_loss = 0

        with torch.no_grad():

            for data in data_loader:

                latent_variable, target = data
                latent_variable, target = latent_variable.cuda(), target.cuda()

                logit = self.model(latent_variable)
                loss = self.criterion(logit, target)
                prediction = logit.argmax(dim=-1)
                batch_size = target.size(0)
                total_samples += batch_size
                correct_samples += (prediction == target).long().sum().item()
                total_loss += batch_size * loss.item()

        loss = total_loss / total_samples
        accuracy = correct_samples / total_samples

        return loss, accuracy

    def get_data(self, output_probability: bool = False) -> List[torch.Tensor]:

        test_dataset = ClassificationDataset(
            sample_path=os.path.join(self.base_path, "vanilla_sample_test.tsv"),
            latent_variable_path=os.path.join(self.base_path, "vanilla_sample_test.npy"),
            target="label"
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )

        self.model.eval()

        latent_variable_list = []
        target_list = []
        probability_list = []

        with torch.no_grad():
            for data in test_loader:
                latent_variable, target = data
                latent_variable, target = latent_variable.cuda(), target.cuda()
                probability = self.model.get_probability(latent_variable=latent_variable, target=target)
                latent_variable_list.append(latent_variable)
                target_list.append(target)
                probability_list.append(probability)

        latent_variable = torch.cat(latent_variable_list, dim=0)
        target = torch.cat(target_list, dim=0)
        probability = torch.cat(probability_list, dim=0)

        output = [latent_variable, target]
        if output_probability:
            output.append(probability)

        return output