import torch
from torch import nn
from src.module.attention.multi_head_attention import MultiHeadAttention
from src.module.layer.position_wise_feed_forward import EncoderPositionWiseFeedForward

class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_size: int, ff_size: int, num_heads: int, dropout: float) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.src_src_attention = MultiHeadAttention(
            num_heads=num_heads,
            size=hidden_size,
            dropout=dropout
        )
        self.feed_forward = EncoderPositionWiseFeedForward(
            hidden_size=hidden_size,
            ff_size=ff_size,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.FloatTensor (batch_size, seq_len, hidden_size)
        :param mask: torch.ByteTensor (batch_size, 1, seq_len)
        :return : torch.FloatTensor (batch_size, seq_len, hidden_size)
        """
        x_norm = self.layer_norm(x)
        h = self.src_src_attention(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o