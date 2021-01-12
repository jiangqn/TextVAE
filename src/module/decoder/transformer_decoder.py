import torch
from torch import nn
from typing import List, Tuple
from src.module.decoder.decoder import Decoder
from src.module.layer.positional_encoding import PositionalEncoding
from src.module.layer.transformer_decoder_layer import TransformerDecoderLayer
from src.utils.subsequent_mask import subsequent_mask
from src.constants import PAD_INDEX

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
            x = layer(x, mask)

        return self.layer_norm(x)

    def decode(self, latent_variable: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param max_len: int
        :return : torch.FloatTensor (batch_size, max_len, vocab_size)
        """
        pass

    def efficient_decode(self, latent_variable: torch.Tensor, max_len) -> torch.Tensor:
        """
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param max_len: int
        :return : torch.FloatTensor (batch_size, max_len, vocab_size)
        """
        pass