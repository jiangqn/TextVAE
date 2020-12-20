import torch
from torch import nn
from src.module.encoder.encoder import Encoder

class TransformerEncoder(Encoder):

    def __init__(self):
        super(TransformerEncoder, self).__init__()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        pass