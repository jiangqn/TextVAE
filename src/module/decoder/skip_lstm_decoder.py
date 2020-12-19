import torch
from torch import nn
from typing import List, Tuple
from src.module.rnn_cell.skip_multi_layer_lstm_cell import SkipMultiLayerLSTMCell
from src.constants import SOS_INDEX, EOS_INDEX
from src.module.decoder.decoder import Decoder

class SkipLSTMDecoder(Decoder):

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, latent_size: int,
                 num_layers: int, dropout: float, word_dropout: float, decoder_generator_tying: bool, initial_hidden_type: str) -> None:
        super(SkipLSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        assert initial_hidden_type in ["zero", "latent_projection"]
        self.initial_hidden_type = initial_hidden_type
        if initial_hidden_type == "latent_projection":
            self.latent_hidden_projection = nn.Linear(latent_size, hidden_size * num_layers)
            self.latent_cell_projection = nn.Linear(latent_size, hidden_size * num_layers)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.word_dropout = word_dropout
        self.rnn_cell = SkipMultiLayerLSTMCell(
            input_size=embed_size,
            hidden_size=hidden_size,
            latent_size=latent_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size)
        )
        self.generator = nn.Linear(embed_size, vocab_size)
        if decoder_generator_tying:
            self.generator.weight = self.embedding.weight

    def _initial_hidden(self, latent_variable: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :return initial_hidden: torch.FloatTensor (num_layers, batch_size, hidden_size)
        """
        batch_size = latent_variable.size(0)
        if self.initial_hidden_type == "zero":
            initial_hidden = torch.zeros(size=(self.num_layers, batch_size, self.hidden_size),
                                         device=latent_variable.device)
            initial_cell = torch.zeros(size=(self.num_layers, batch_size, self.hidden_size),
                                       device=latent_variable.device)
        else:
            initial_hidden = self.latent_hidden_projection(
                latent_variable)  # torch.FloatTensor (batch_size, num_layers, hidden_size)
            initial_hidden = initial_hidden.reshape(batch_size, self.num_layers, self.hidden_size).transpose(0, 1)
            initial_cell = self.latent_cell_projection(latent_variable)
            initial_cell = initial_cell.reshape(batch_size, self.num_layers, self.hidden_size).transpose(0, 1)
        return initial_hidden, initial_cell

    def forward(self, latent_variable: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param trg: torch.LongTensor (batch_size, seq_len)
        :return logit: torch.FloatTensor (batch_size, seq_len, vocab_size)
        """

        if self.training:
            trg = self._word_dropout(trg)
        # trg = self._word_dropout(trg)

        hidden, cell = self._initial_hidden(latent_variable)

        max_len = trg.size(1)
        logit = []
        for i in range(max_len):
            hidden, cell, token_logit = self.step(hidden, cell, trg[:, i], latent_variable)
            logit.append(token_logit)
        logit = torch.stack(logit, dim=1)
        return logit

    def decode(self, latent_variable: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :param max_len: int
        :return logit: torch.FloatTensor (batch_size, max_len, vocab_size)
        """

        hidden, cell = self._initial_hidden(latent_variable)

        batch_size = latent_variable.size(0)
        token = torch.tensor([SOS_INDEX] * batch_size, dtype=torch.long, device=latent_variable.device)
        logit = []
        for i in range(max_len):
            hidden, cell, token_logit = self.step(hidden, cell, token, latent_variable)
            if i == 0:
                token_logit[:, EOS_INDEX] = 0
            token = token_logit.argmax(dim=-1)
            logit.append(token_logit)
        logit = torch.stack(logit, dim=1)
        return logit

    def step(self, hidden: torch.Tensor, cell: torch.Tensor, token: torch.Tensor, latent_variable: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param hidden: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :param cell: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :param token: torch.LongTensor (batch_size,)
        :param latent_variable: torch.FloatTensor (batch_size, latent_size)
        :return hidden: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :return token_logit: torch.FloatTensor (batch_size, logit)
        """
        token_embedding = self.embedding(token.unsqueeze(0)).squeeze(0) # torch.FloatTensor (batch_size, embed_size)
        hidden, cell = self.rnn_cell(torch.cat((token_embedding, latent_variable), dim=1), (hidden, cell))
        top_hidden = hidden[-1]
        output = self.output_projection(top_hidden)
        token_logit = self.generator(output)
        return hidden, cell, token_logit