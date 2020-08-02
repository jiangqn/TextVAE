import torch
from torch import nn
import torch.nn.functional as F

class TextCNN(nn.Module):

    def __init__(self, vocab_size, embed_size, kernel_sizes, kernel_num, dropout, num_categories):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.convs = nn.ModuleList(
            nn.Conv1d(
                in_channels=embed_size,
                out_channels=kernel_num,
                kernel_size=kernel_size
            ) for kernel_size in kernel_sizes
        )
        self.dropout = dropout
        self.feature_size = len(kernel_sizes) * kernel_num
        self.linear = nn.Linear(self.feature_size, num_categories)

    def forward(self, sentence):
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