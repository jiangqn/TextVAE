import torch
from torch import nn
import torch.nn.functional as F

class LanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, weight_tying):
        super(LanguageModel, self).__init__()
        self.dropout = dropout
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            dropout=self.dropout,
            batch_first=True
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, embed_size)
        )
        self.generator = nn.Linear(embed_size, vocab_size)
        if weight_tying:
            self.generator.weight = self.embedding.weight

    def forward(self, sentence):
        sentence = self.embedding(sentence)
        sentence = F.dropout(sentence, p=self.dropout, training=self.training)
        hidden, _ = self.rnn(sentence)
        output = self.output_projection(hidden)
        logit = self.generator(output)
        return logit