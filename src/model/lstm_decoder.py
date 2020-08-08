import torch
from torch import nn
from src.model.multi_layer_lstm_cell import MultiLayerLSTMCell
from src.constants import SOS_INDEX

class Decoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, weight_tying):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn_cell = MultiLayerLSTMCell(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size)
        )
        self.generator = nn.Linear(embed_size, vocab_size)
        if weight_tying:
            self.generator.weight = self.embedding.weight

    def forward(self, hidden, trg):
        cell = torch.zeros(hidden.size(), dtype=hidden.dtype, device=hidden.device)
        max_len = trg.size(1)
        logit = []
        for i in range(max_len):
            (hidden, cell), token_logit = self.step((hidden, cell), trg[:, i])
            logit.append(token_logit)
        logit = torch.stack(logit, dim=1)
        return logit

    def step(self, states, token):
        hidden, cell = states
        token_embedding = self.embedding(token.unsqueeze(0)).squeeze(0)
        hidden, cell = self.rnn_cell(token_embedding, (hidden, cell))
        top_hidden = hidden[-1]
        output = self.output_projection(top_hidden)
        token_logit = self.generator(output)
        return (hidden, cell), token_logit

    def decode(self, hidden, max_len):
        cell = torch.zeros(hidden.size(), dtype=hidden.dtype, device=hidden.device)
        batch_size = hidden.size(1)
        token = torch.tensor([SOS_INDEX] * batch_size, dtype=torch.long, device=hidden.device)
        logit = []
        for i in range(max_len):
            (hidden, cell), token_logit = self.step((hidden, cell), token)
            token = token_logit.argmax(dim=-1)
            logit.append(token_logit)
        logit = torch.stack(logit, dim=1)
        return logit