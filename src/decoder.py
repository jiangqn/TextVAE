import torch
from torch import nn
from src.multi_layer_gru_cell import MultiLayerGRUCell

class Decoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, weight_tying):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn_cell = MultiLayerGRUCell(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.embed_size)
        )
        self.generator = nn.Linear(self.embed_size, vocab_size)
        if weight_tying:
            self.generator.weight = self.embedding.weight

    def forward(self, encoding, trg):
        pass