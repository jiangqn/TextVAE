import torch
from torch import nn
from src.module.attention.multi_head_attention import MultiHeadAttention
from src.module.layer.position_wise_feed_forward import EncoderPositionWiseFeedForward

class TransformerDecoderLayer(nn.Module):

    def __init__(self, hidden_size: int, ff_size: int, num_heads: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()

    def forward(self, x: torch.Tensor, latent_variable: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.FloatTensor (batch_size, seq_len, hidden_size)
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param mask: torch.ByteTensor (batch_size, 1, seq_len)
        """
        pass