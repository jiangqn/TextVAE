import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from src.module.encoder.encoder import Encoder
from src.module.decoder.decoder import Decoder
from typing import Tuple
import numpy as np
from src.utils.generate_pad import generate_pad
from src.constants import PAD_INDEX

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
            if hasattr(encoder, "embedding_projection") and hasattr(decoder, "embedding_projection"):
                self.encoder.embedding_projection.weight = self.decoder.embedding_projection.weight

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

    def gradient_encode(self, sentence: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        :param sentence: torch.LongTensor (batch_size, seq_len)
        :return latent_variable: torch.FloatTensor (batch_size, latent_size)
        """

        lr = kwargs.get('lr', 0.3)
        weight_deacy = kwargs.get('weight_decay', 0)
        max_iter = kwargs.get('max_iter', 100)

        src = sentence[:, 1:]
        trg_input = sentence
        batch_size = sentence.size(0)
        pad = generate_pad(size=(batch_size, 1), device=sentence.device)
        trg_output = torch.cat((sentence[:, 1:], pad), dim=-1)

        posterior_mean, posterior_std = self.encode(src)
        latent_variable = nn.Parameter(posterior_mean)

        optimizer = optim.Adam([latent_variable], lr=lr, weight_decay=weight_deacy)

        criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

        min_loss = 1e9
        best_latent_variable = latent_variable.data
        best_i = 0

        for i in range(max_iter):

            optimizer.zero_grad()

            logit = self.decoder(latent_variable, trg_input)
            trg_output = trg_output.view(-1)
            output_size = logit.size(-1)
            logit = logit.view(-1, output_size)

            loss = criterion(logit, trg_output)
            loss.backward()

            optimizer.step()

            mask = (trg_output != PAD_INDEX)
            prediction = logit.argmax(dim=-1)
            accuracy = (prediction.masked_select(mask) == trg_output.masked_select(mask)).float().mean().item()

            if loss.item() < min_loss:
                min_loss = loss.item()
                best_latent_variable = latent_variable.data
                best_i = i
                # print(i, accuracy)

        # print(best_latent_variable.std(dim=1))

        return best_latent_variable

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
        assert "max_len" in kwargs
        if "num" in kwargs:
            device = self.encoder.embedding.weight.device
            # U = torch.randn(size=(kwargs["num"], self.latent_size), device=device)
            # V = U.matmul(self.aggregated_posterior_weight) + self.aggregated_posterior_mean
            # latent_variable = (1 - lambd) * U + lambd * V
            latent_variable = torch.randn(size=(kwargs["num"], self.latent_size), device=device)
        else:
            latent_variable = kwargs["latent_variable"]
        max_len = kwargs["max_len"]

        self.eval()
        with torch.no_grad():
            logit = self.decoder.efficient_decode(latent_variable, max_len)
        output = logit.argmax(dim=-1)
        return_list = [output]
        if kwargs.get("output_latent_variable", False):
            latent_variable = latent_variable.cpu().numpy()
            return_list.append(latent_variable)
        if kwargs.get("output_logit", False):
            return_list.append(logit)
        return return_list[0] if len(return_list) == 1 else return_list