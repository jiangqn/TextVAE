import torch
from torch import nn
from typing import List, Tuple
from src.model.multi_layer_gru_cell import MultiLayerGRUCell
from src.constants import SOS_INDEX, EOS_INDEX

class Decoder(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, dropout: float, weight_tying: bool) -> None:
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
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size)
        )
        self.generator = nn.Linear(embed_size, vocab_size)
        if weight_tying:
            self.generator.weight = self.embedding.weight

    def forward(self, hidden: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        '''
        :param hidden: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :param trg: torch.LongTensor (batch_size, seq_len)
        :return logit: torch.FloatTensor (batch_size, seq_len, vocab_size)
        '''

        max_len = trg.size(1)
        logit = []
        for i in range(max_len):
            hidden, token_logit = self.step(hidden, trg[:, i])
            logit.append(token_logit)
        logit = torch.stack(logit, dim=1)
        return logit

    def step(self, hidden: torch.Tensor, token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        :param hidden: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :param token: torch.LongTensor (batch_size,)
        :return hidden: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :return token_logit: torch.FloatTensor (batch_size, logit)
        '''
        token_embedding = self.embedding(token.unsqueeze(0)).squeeze(0)
        hidden = self.rnn_cell(token_embedding, hidden)
        top_hidden = hidden[-1]
        output = self.output_projection(top_hidden)
        token_logit = self.generator(output)
        return hidden, token_logit

    def decode(self, hidden: torch.Tensor, max_len: int) -> torch.Tensor:
        '''
        :param hidden: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :param max_len: int
        :return logit: torch.FloatTensor (batch_size, seq_len, vocab_size)
        '''
        batch_size = hidden.size(1)
        token = torch.tensor([SOS_INDEX] * batch_size, dtype=torch.long, device=hidden.device)
        logit = []
        for i in range(max_len):
            hidden, token_logit = self.step(hidden, token)
            if i == 0:
                try:
                    token_logit[:, EOS_INDEX] = 0
                except:
                    print(hidden.size())
                    print(token_logit.size())
            token = token_logit.argmax(dim=-1)
            logit.append(token_logit)
        logit = torch.stack(logit, dim=1)
        return logit