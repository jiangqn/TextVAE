import torch
from torch import nn
from torch.nn.init import constant_
from src.constants import PAD_INDEX, UNK_INDEX, SOS_INDEX, EOS_INDEX

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

    def _word_dropout(self, trg: torch.Tensor) -> torch.Tensor:
        """
        :param trg: torch.LongTensor (batch_size, seq_len)
        :return mask * trg: torch.LongTensor (batch_size, seq_len)
        """

        pad_mask = (trg == PAD_INDEX) | (trg == SOS_INDEX) | (trg == EOS_INDEX)
        p = torch.FloatTensor(trg.size()).to(trg.device)
        constant_(p, 1 - self.word_dropout)
        mask = torch.bernoulli(p).long()
        masked_trg = mask * trg + (1 - mask) * UNK_INDEX
        masked_trg.masked_fill_(pad_mask, PAD_INDEX)
        return masked_trg