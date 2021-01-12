import torch
from torch import nn
from typing import List, Tuple
from src.module.decoder.decoder import Decoder
from src.module.layer.positional_encoding import PositionalEncoding
from src.module.layer.transformer_decoder_layer import TransformerDecoderLayer
from src.utils.subsequent_mask import subsequent_mask
from src.constants import PAD_INDEX, SOS_INDEX, EOS_INDEX

class TransformerDecoder(Decoder):

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, latent_size: int, ff_size: int,
                 num_layers: int, num_heads: int, dropout: float, word_dropout: float, decoder_generator_tying: bool) -> None:
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.pe = PositionalEncoding(hidden_size, max_len=100)
        self.embedding_projection = nn.Linear(embed_size, hidden_size, bias=False)
        self.embedding_dropout = nn.Dropout(dropout)
        self.word_dropout = word_dropout
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_size=hidden_size,
                latent_size=latent_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.generator = nn.Linear(hidden_size, vocab_size)
        if decoder_generator_tying:
            self.generator.weight = self.embedding.weight

    def forward(self, latent_variable: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param trg: torch.LongTensor (batch_size, seq_len)
        :return logit: torch.FloatTensor (batch_size, seq_len, vocab_size)
        """

        if self.training:
            trg = self._word_dropout(trg)

        batch_size, seq_len = trg.size()

        mask = (trg != PAD_INDEX).unsqueeze(1)
        mask = mask & subsequent_mask(seq_len).type_as(mask)

        x = self.embedding(trg)
        x = self.pe(x)
        x = self.embedding_projection(x)
        x = self.embedding_dropout(x)

        for layer in self.layers:
            x = layer(x, latent_variable, mask)

        x = self.layer_norm(x)
        logit = self.generator(x)

        return logit

    def decode(self, latent_variable: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param max_len: int
        :return : torch.FloatTensor (batch_size, max_len, vocab_size)
        """
        batch_size = latent_variable.size(0)
        trg = torch.tensor([SOS_INDEX] * batch_size, dtype=torch.long, device=latent_variable.device).unsqueeze(-1)
        logit = []
        for i in range(max_len):
            token_logit = self.forward(latent_variable, trg)[:, -1] # (batch_size, vocab_size)
            if i == 0:
                token_logit[:, EOS_INDEX] = 0
            token = token_logit.argmax(dim=-1, keepdim=True)  # (batch_size, 1)
            trg = torch.cat([trg, token], dim=-1)
            logit.append(token_logit)
        logit = torch.stack(logit, dim=1)
        return logit

    def efficient_decode(self, latent_variable: torch.Tensor, max_len) -> torch.Tensor:
        """
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param max_len: int
        :return : torch.FloatTensor (batch_size, max_len, vocab_size)
        """
        batch_size = latent_variable.size(0)
        mask = subsequent_mask(max_len).byte().to(latent_variable.device)

        trg = torch.tensor([SOS_INDEX] * batch_size, dtype=torch.long, device=latent_variable.device).unsqueeze(-1)
        trg_embedding_memory = self.embedding(trg)
        trg_embedding_memory = self.pe.efficient_forward(trg_embedding_memory, 0)
        trg_embedding_memory = self.embedding_projection(trg_embedding_memory)
        trg_embedding_memory = self.embedding_dropout(trg_embedding_memory)
        x_memories = []
        x = trg_embedding_memory
        for layer in self.layers:
            x = layer(x, mask[:, 0:1, :])

        logit = []
        for i in range(max_len):
            pass