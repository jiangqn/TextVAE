import torch
from torch import nn
import torch.nn.functional as F
from src.module.encoder.encoder import Encoder
from src.module.decoder.decoder import Decoder
from typing import Tuple

class TextVAE(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, latent_size: int, encoder_decoder_tying: bool) -> None:
        super(TextVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_output_size = self.encoder.output_size
        self.latent_size = latent_size
        self.posterior_mean_projection = nn.Linear(self.encoder_output_size, self.latent_size)
        self.posterior_std_projection = nn.Linear(self.encoder_output_size, self.latent_size)
        if encoder_decoder_tying:
            self.encoder.embedding.weight = self.decoder.embedding.weight

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param src: torch.LongTensor (batch_size, src_seq_len)
        :param trg: torch.LongTensor (batch_size, trg_seq_len)
        :return logit: torch.FloatTensor (batch_size, trg_seq_len vocab_size)
        :return posterior_mean: torch.FloatTensor (batch_size, latent_size)
        :return posterior_std: torch.FloatTensor (batch_size, latent_size)
        """

        posterior_mean, posterior_std = self.encode(src)
        if self.training:
            latent_variable = self.reparametrize(posterior_mean, posterior_std)
        else:
            latent_variable = posterior_mean

        logit = self.decoder(latent_variable, trg)
        return logit, posterior_mean, posterior_std

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param src: torch.LongTensor (batch_size, seq_len)
        :return posterior_mean: torch.FloatTensor (batch_size, latent_size)
        :return posterior_std: torch.FloatTensor (batch_size, latent_size)
        """

        encoder_representation = self.encoder(src)
        posterior_mean = self.posterior_mean_projection(encoder_representation)
        posterior_std = torch.exp(self.posterior_std_projection(encoder_representation))
        return posterior_mean, posterior_std

    def reparametrize(self, posterior_mean: torch.Tensor, posterior_std: torch.Tensor) -> torch.Tensor:
        """
        :param posterior_mean: torch.FloatTensor (batch_size, latent_size)
        :param posterior_std: torch.FloatTensor (batch_size, latent_size)
        :return latent_variable: torch.FloatTensor (batch_size, latent_size)
        """

        noise = torch.randn(size=posterior_mean.size(), device=posterior_mean.device)
        latent_variable = posterior_mean + posterior_std * noise
        return latent_variable