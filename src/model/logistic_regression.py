import torch
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
import numpy as np
from typing import Tuple
from src.utils.classification_dataset import get_classification_dataloader

class LogisticRegression(nn.Module):

    def __init__(self, hidden_size: int, num_categories: int) -> None:
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(hidden_size, num_categories)

    def forward(self, latent_code: torch.Tensor) -> torch.Tensor:
        '''
        :param latent_code: torch.FloatTensor (batch_size, hidden_size)
        :return logit: torch.FloatTensor (batch_size, num_categories)
        '''

        logit = self.linear(latent_code)
        return logit

class LogisticRegressionClassifier(object):

    def __init__(self, lr: float = 0.001, batch_size: int = 64, weight_decay: float = 0, max_epoches: int = 10, max_patience: int = 10) -> None:
        super(LogisticRegressionClassifier, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.max_epoches = max_epoches
        self.max_patience = max_patience
        self.model = None
        self.best_model = None

    def fit(self, feature: np.ndarray, label: np.ndarray) -> None:
        '''
        :param feature: np.ndarray (num, hidden_size)
        :param label: np.ndarray (num,)
        '''

        hidden_size = feature.shape[1]
        num_categories = int(label.max()) + 1
        train_loader, dev_loader = get_classification_dataloader(feature, label, batch_size=self.batch_size, shuffle=True, split=True)
        self.model = LogisticRegression(hidden_size=hidden_size, num_categories=num_categories).cuda()

        self.criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        min_loss = 1e9
        patience = 0
        early_stop = False

        global_step = 0

        for epoch in range(self.max_epoches):

            for i, data in enumerate(train_loader):

                self.model.train()
                optimizer.zero_grad()

                feature, label = data
                feature, label = feature.cuda(), label.cuda()

                logit = self.model(feature)
                loss = self.criterion(logit, label)
                loss.backward()
                optimizer.step()

                if global_step % 100 == 0:
                    dev_loss, dev_accuracy = self._eval(dev_loader)
                    if dev_loss < min_loss:
                        min_loss = dev_loss
                        self.best_model = self.model
                        patience = 0
                    else:
                        patience += 1
                        if patience >= self.max_patience:
                            early_stop = True
                            break

                global_step += 1

            if patience >= self.max_patience:
                break

        if not early_stop:
            print('not converge, increase max_epoches')

        self.model = self.best_model
        self.best_model = None

    def predict(self, feature: np.ndarray, output_probability: bool = False):
        feature = torch.from_numpy(feature).float().cuda()

        self.model.eval()
        with torch.no_grad():
            logit = self.model(feature)
            prediction = logit.argmax(dim=-1).cpu().numpy()
            probability = torch.softmax(logit, dim=-1).max(dim=-1, keepdim=False)[0].cpu().numpy()

        if output_probability:
            return prediction, probability
        else:
            return prediction


    def eval(self, feature: np.ndarray, label: np.ndarray) -> float:
        prediction = self.predict(feature)
        accuracy = float((prediction == label).mean())
        return accuracy

    def _eval(self, dataloader: DataLoader) -> Tuple[float, float]:

        total_samples = 0
        correct_samples = 0
        total_loss = 0

        self.model.eval()
        with torch.no_grad():

            for data in dataloader:
                feature, label = data
                feature, label = feature.cuda(), label.cuda()
                logit = self.model(feature)
                loss = self.criterion(logit, label)

                batch_size = feature.size(0)
                total_samples += batch_size
                total_loss += batch_size * loss.item()
                prediction = logit.argmax(dim=-1)
                correct_samples += (prediction == label).long().sum().item()

        loss = total_loss / total_samples
        accuracy = correct_samples / total_samples
        return loss, accuracy