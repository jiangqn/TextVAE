import torch
from torch import nn
from src.encoder import Encoder
from src.decoder import Decoder

class TextVAE(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, enc_dec_tying, dec_gen_tying):
        super(TextVAE, self).__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            weight_tying=dec_gen_tying
        )
        if enc_dec_tying:
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(self, src, trg):
        encoding = self.encoder(src)
        return encoding

    def sample(self):
        pass