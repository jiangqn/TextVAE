import torch
from torch import nn
from src.module.encoder.encoder import Encoder
from src.module.layer.positional_encoding import PositionalEncoding
from src.module.layer.transformer_encoder_layer import TransformerEncoderLayer
from src.module.attention.attention import get_attention
from src.constants import PAD_INDEX

class TransformerEncoder(Encoder):

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, ff_size: int,
                 num_layers: int, num_heads: int, dropout: int) -> None:
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.pe = PositionalEncoding(embed_size, max_len=100)
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
        self.attention = get_attention(
            query_size=hidden_size,
            key_size=hidden_size,
            attention_type="Bilinear"
        )
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.hidden_size = hidden_size

    @property
    def output_size(self) -> int:
        return self.hidden_size

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        :param src: torch.LongTensor (batch_size, seq_len)
        :return : torch.FloatTensor (batch_size, hidden_size)
        """
        mask = (src != PAD_INDEX).unsqueeze(1)  # torch.ByteTensor (batch_size, 1, seq_len)

        x = self.embedding(src)
        x = self.pe(x)
        x = self.embedding_projection(x)
        x = self.embedding_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.layer_norm(x)

        # x_pooling = x.max(dim=1)[0]
        float_mask = mask.transpose(1, 2).float()
        x_pooling = (x * float_mask).sum(dim=1, keepdim=False) / float_mask.sum(dim=1, keepdim=False)
        mask = mask.squeeze(1)
        x_pooling = self.attention(x_pooling, x, x, mask)
        x_pooling = self.attention_layer_norm(x_pooling)
        return x_pooling