import torch
from torch import nn
from src.module.encoder.encoder import Encoder

class TransformerEncoder(Encoder):

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, ff_size: int,
                 num_layers: int, num_heads: int, dropout: int) -> None:
        super(TransformerEncoder, self).__init__()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        pass