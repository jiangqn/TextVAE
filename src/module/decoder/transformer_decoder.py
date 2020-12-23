import torch
from torch import nn
from typing import List, Tuple
from src.module.decoder.decoder import Decoder

class TransformerDecoder(Decoder):

    def __init__(self):
        super(TransformerDecoder, self).__init__()

    def forward(self, latent_variable: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        pass

    def decode(self, latent_variable: torch.Tensor, max_len: int) -> torch.Tensor:
        pass