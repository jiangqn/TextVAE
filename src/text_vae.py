import torch
from torch import nn
import torch.nn.functional as F
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
        self.mean_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.std_projection = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, src, trg):
        encoding, _, _ = self.encode(src)
        return self.decoder(encoding, trg)

    def encode(self, src):
        final_states = self.encoder(src)
        mean = self.mean_projection(final_states)
        std = F.softplus(self.std_projection(final_states))
        sample = torch.randn(size=mean.size(), device=mean.device)
        encoding = mean + std * sample
        return encoding, mean, std