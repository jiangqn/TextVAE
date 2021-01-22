import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from src.analyze.dataset import ClassificationDataset

class LogisticRegression(nn.Module):

    def __init__(self, latent_size: int, output_size: int) -> None:
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=latent_size, out_features=output_size)

    def forward(self, latent_variable: torch.Tensor) -> torch.Tensor:
        """
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :return logit: torch.FloatTensor (batch_size, output_size)
        """
        logit = self.linear(latent_variable)
        return logit

    def get_probability(self, latent_variable: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param target: torch.LongTensor (batch_size,)
        :return prob: torch.FloatTensor (batch,)
        """
        logit = self.linear(latent_variable)
        prob = torch.softmax(logit, dim=-1)
        prob = torch.gather(prob, 1, target.unsqueeze(-1)).squeeze(-1)
        return prob