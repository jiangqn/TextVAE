import torch
from torch import nn
import torch.nn.functional as F
from typing import List
import numpy as np

class TextCNN(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, kernel_sizes: List[int], kernel_num: int, dropout: float, num_categories: int) -> None:
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.convs = nn.ModuleList(
            nn.Conv1d(
                in_channels=embed_size,
                out_channels=kernel_num,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ) for kernel_size in kernel_sizes
        )
        self.dropout = dropout
        self.feature_size = len(kernel_sizes) * kernel_num
        self.linear = nn.Linear(self.feature_size, num_categories)

    def load_pretrained_embeddings(self, **kwargs) -> None:
        assert ("path" in kwargs) ^ ("embedding" in kwargs)
        if "path" in kwargs:
            embedding = np.load(kwargs["path"])
        else:
            embedding = kwargs["embedding"]
        self.embedding.weight.data.copy_(torch.tensor(embedding))

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        :param sentence: torch.LongTensor (batch_size, seq_len)
        :return: torch.FloatTensor (batch_size, num_categories)
        """

        embedding = self.embedding(sentence)
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        embedding = embedding.transpose(1, 2)
        output = torch.cat([
            torch.max(
                F.relu(conv(embedding)),
                dim=-1
            )[0] for conv in self.convs
        ], dim=1)
        output = F.dropout(output, p=self.dropout, training=self.training)
        logit = self.linear(output)
        return logit