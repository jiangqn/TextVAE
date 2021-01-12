import torch
from torch import nn
from src.module.encoder.encoder import Encoder
from src.module.layer.positional_encoding import PositionalEncoding
from src.module.layer.transformer_encoder_layer import TransformerEncoderLayer
from src.constants import PAD_INDEX

class TransformerEncoder(Encoder):

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, ff_size: int,
                 num_layers: int, num_heads: int, dropout: int) -> None:
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.pe = PositionalEncoding(hidden_size, max_len=100)
        self.embedding_projection = nn.Linear(embed_size, hidden_size, bias=False)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        :param src: torch.LongTensor (batch_size, seq_len)
        :return : torch.FloatTensor (batch_size, seq_len, hidden_size)
        """
        mask = (src != PAD_INDEX).unsqueeze(1)  # torch.ByteTensor (batch_size, 1, seq_len)

        x = self.embedding(src)
        x = self.pe(x)
        x = self.embedding_projection(x)
        x = self.embedding_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x)