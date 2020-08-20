import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class LanguageModel(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, dropout: float, weight_tying: bool) -> None:
        super(LanguageModel, self).__init__()
        self.dropout = dropout
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
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

    def load_pretrained_embeddings(self, **kwargs) -> None:
        assert ('path' in kwargs) ^ ('embedding' in kwargs)
        if 'path' in kwargs:
            embedding = np.load(kwargs['path'])
        else:
            embedding = kwargs['embedding']
        self.embedding.weight.data.copy_(torch.tensor(embedding))

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        '''
        :param sentence: torch.LongTensor (batch_size, seq_len)
        :return logit: torch.FloatTensor (batch_size, seq_len, vocab_size)
        '''

        sentence = self.embedding(sentence)
        sentence = F.dropout(sentence, p=self.dropout, training=self.training)
        hidden, _ = self.rnn(sentence)
        output = self.output_projection(hidden)
        logit = self.generator(output)
        return logit