import torch
from torch import nn
import torch.nn.functional as F
from src.module.encoder.encoder import Encoder
from src.module.decoder.decoder import Decoder
from typing import Tuple
import numpy as np

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

    def load_pretrained_embeddings(self, **kwargs) -> None:
        assert ("path" in kwargs) ^ ("embedding" in kwargs)
        if "path" in kwargs:
            embedding = np.load(kwargs["path"])
        else:
            embedding = kwargs["embedding"]
        self.encoder.embedding.weight.data.copy_(torch.tensor(embedding))
        self.decoder.embedding.weight.data.copy_(torch.tensor(embedding))

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
    
    def sample(self, **kwargs):
        """
        :num: int
        :latent_variable: torch.FloatTensor (sample_num, latent_size)
        :output: torch.LongTensor (sample_num, max_len)
        :logit: torch.FloatTensor (sample_num, max_len, vocab_size)
        """

        assert ("num" in kwargs) ^ ("latent_variable" in kwargs)
        if "num" in kwargs:
            latent_variable = torch.randn(size=(kwargs["num"], self.latent_size),
                                   device=self.encoder.embedding.weight.device)
        else:
            latent_variable = kwargs["latent_variable"]
        max_len = kwargs.get("max_len", 15)
        logit = self.decoder.decode(latent_variable, max_len)
        output = logit.argmax(dim=-1)
        return_list = [output]
        if kwargs.get("output_latent_variable", False):
            latent_variable = latent_variable.cpu().numpy()
            return_list.append(latent_variable)
        if kwargs.get("output_logit", False):
            return_list.append(logit)
        return return_list[0] if len(return_list) == 1 else return_list