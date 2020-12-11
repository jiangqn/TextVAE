import torch
from torch import nn
import torch.nn.functional as F
from src.module.encoder.encoder import Encoder
from src.module.decoder.decoder import Decoder
from typing import Tuple

class TextVAE(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, latent_size: int) -> None:
        super(TextVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size
        self.mean_projection = nn.Linear(self.encoder.output_size, self.latent_size)
        self.std_projection = nn.Linear(self.encoder.output_size, self.latent_size)

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        if self.training:
            latent_encoding, mean, std = self.probabilistic_encode(src)
        else:
            mean, std = self.encode(src)
            latent_encoding = mean
        logit = self.decoder(latent_encoding, trg)
        return logit, mean, std

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        :param src: torch.LongTensor (batch_size, seq_len)
        :return mean: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :return std: torch.FloatTensor (num_layers, batch_size, hidden_size)
        '''

        encoder_representation = self.encoder(src)
        mean = self.mean_projection(encoder_representation)
        # std = F.softplus(self.std_projection(final_states))
        std = torch.exp(self.std_projection(encoder_representation))
        return mean, std

    def reparametrize(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        '''
        :param mean: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :param std: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :return encoding: torch.FloatTensor (num_layers, batch_size, hidden_size)
        '''

        noise = torch.randn(size=mean.size(), device=mean.device)
        encoding = mean + std * noise
        return encoding

    def probabilistic_encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        :param src: torch.LongTensor (batch_size, seq_len)
        :return encoding: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :return mean: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :return std: torch.FloatTensor (num_layers, batch_size, hidden_size)
        '''

        mean, std = self.encode(src)
        encoding = self.reparametrize(mean, std)
        return encoding, mean, std