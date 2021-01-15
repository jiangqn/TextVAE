import torch
from torch import nn

class ConditionalVAE(nn.Module):

    def __init__(self):
        super(ConditionalVAE, self).__init__()

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        pass