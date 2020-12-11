import torch
from torch import nn
import torch.nn.functional as F
from src.module.encoder.encoder import Encoder
from src.constants import PAD_INDEX

class BOWMLPEncoder(Encoder):

    """
    The bag-of-words mlp encoder of TextVAE
    """

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, dropout: float) -> None:
        super(BOWMLPEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        self.dropout = dropout

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: torch.LongTensor (batch_size, seq_len)
        representation: torch.FloatTensor (batch_size, hidden_size)
        """
        mask = (src != PAD_INDEX).float()
        src_len = mask.sum(dim=1, keepdim=True)
        embedding = self.embedding(src)
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        pooled_embedding = (embedding * mask.unsqueeze(-1)).sum(dim=1, keepdim=False) / src_len
        pooled_embedding = F.dropout(pooled_embedding, p=self.dropout, training=self.training)
        representation = self.mlp(pooled_embedding)
        return representation