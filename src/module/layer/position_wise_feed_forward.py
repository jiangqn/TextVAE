import torch
from torch import nn

class EncoderPositionWiseFeedForward(nn.Module):

    def __init__(self, hidden_size: int, ff_size: int, dropout: float) -> None:
        super(EncoderPositionWiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.FloatTensor (batch_size, seq_len, hidden_size)
        :return : torch.FloatTensor (batch_size, seq_len, hidden_size)
        """
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x

class DecoderPositionWiseFeedForward(nn.Module):

    def __init__(self, hidden_size: int, latent_size: int, ff_size: int, dropout: float) -> None:
        super(DecoderPositionWiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(hidden_size + latent_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, latent_variable: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.FloatTensor (batch_size, seq_len, hidden_size)
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :return : torch.FloatTensor (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.size()
        latent_size = latent_variable.size(-1)
        x_norm = self.layer_norm(x)
        latent_variable = latent_variable.unsqueeze(1).expand(batch_size, seq_len, latent_size)
        x_cat = torch.cat([x_norm, latent_variable], dim=-1)
        return self.pwff_layer(x_cat) + x