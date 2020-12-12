import torch
from torch import nn

class Decoder(nn.Module):

    """
    The base class of decoder of TextVAE
    """

    def __init__(self) -> None:
        super(Decoder, self).__init__()

    def forward(self, latent_variable: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param trg: torch.LongTensor (batch_size, seq_len)
        :return logit: torch.FloatTensor (batch_size, seq_len, vocab_size)
        """
        raise NotImplementedError("The forward method in class Decoder is not implemented.")

    def decode(self, latent_variable: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param max_len: int
        :return logit: torch.FloatTensor (batch_size, max_len, vocab_size)
        """
        raise NotImplementedError("The decode method in class Decoder is not implemented.")

    def beam_decode(self, latent_variable: torch.Tensor, max_len: int, beam_size: int) -> torch.Tensor:
        """
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param max_len: int
        :return logit: torch.FloatTensor (batch_size, max_len, vocab_size)
        """
        raise NotImplementedError("The beam_decode method in class Decoder is not implemented.")