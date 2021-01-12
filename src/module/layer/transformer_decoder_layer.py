import torch
from torch import nn
from src.module.attention.multi_head_attention import MultiHeadAttention
from src.module.layer.position_wise_feed_forward import DecoderPositionWiseFeedForward

class TransformerDecoderLayer(nn.Module):

    def __init__(self, hidden_size: int, latent_size: int, ff_size: int, num_heads: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.trg_trg_attention = MultiHeadAttention(
            num_heads=num_heads,
            size=hidden_size,
            dropout=dropout
        )
        self.feed_forward = DecoderPositionWiseFeedForward(
            hidden_size=hidden_size,
            latent_size=latent_size,
            ff_size=ff_size,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, latent_variable: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.FloatTensor (batch_size, seq_len, hidden_size)
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param mask: torch.ByteTensor (batch_size, 1, seq_len)
        :return : torch.FloatTensor (batch_size, seq_len, hidden_size)
        """
        x_norm = self.layer_norm(x)
        h = self.trg_trg_attention(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h, latent_variable)
        return o

    def efficient_forward(self, x: torch.Tensor, x_memory: torch.Tensor, latent_variable: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.FloatTensor (batch_size, 1, hidden_size)
        :param x_memory: torch.FloatTensor (batch_size, seq_len, hidden_size)
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param mask: torch.ByteTensor (batch_size, 1, seq_len)
        :return : torch.FloatTensor (batch_size, 1, hidden_size)
        """
        x_norm = self.layer_norm(x)
        x_memory_norm = self.layer_norm(x_memory)
        h = self.trg_trg_attention(x_norm, x_memory_norm, x_memory_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h, latent_variable)
        return o