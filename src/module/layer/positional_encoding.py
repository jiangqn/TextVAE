import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.
    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """
    def __init__(self, size: int = 0, max_len: int = 5000) -> None:
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(size))
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, size, 2, dtype=torch.float) *
                              -(math.log(10000.0) / size)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super().__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, :emb.size(1)]

    def efficient_forward(self, emb: torch.Tensor, index: int) -> torch.Tensor:
        """
        :param emb: torch.FloatTensor (batch_size, 1, size)
        :param index: int
        :return : torch.FloatTensor (batch_size, 1, size)
        """
        return emb + self.pe[:, index: index + 1]