import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import constant_
import numpy as np
from typing import Tuple
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.constants import PAD_INDEX, UNK_INDEX

class TextVAE(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, dropout: float,
                 word_dropout: float, enc_dec_tying: bool, dec_gen_tying: bool) -> None:
        super(TextVAE, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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
        self.word_dropout = word_dropout
        if enc_dec_tying:
            self.decoder.embedding.weight = self.encoder.embedding.weight
        self.mean_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.std_projection = nn.Linear(2 * hidden_size, hidden_size)

    def load_pretrained_embeddings(self, **kwargs) -> None:
        assert ('path' in kwargs) ^ ('embedding' in kwargs)
        if 'path' in kwargs:
            embedding = np.load(kwargs['path'])
        else:
            embedding = kwargs['embedding']
        self.encoder.embedding.weight.data.copy_(torch.tensor(embedding))
        self.decoder.embedding.weight.data.copy_(torch.tensor(embedding))

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        :param src: torch.LongTensor (batch_size, src_seq_len)
        :param trg: torch.LongTensor (batch_size, trg_seq_len)
        :return logit: torch.FloatTensor (batch_size, trg_seq_len, vocab_size)
        :return mean: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :return std: torch.FloatTensor (num_layers, batch_size, hidden_size)
        '''

        encoding, mean, std = self.probabilistic_encode(src)
        trg = self._word_dropout(trg)
        logit = self.decoder(encoding, trg)
        return logit, mean, std

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        :param src: torch.LongTensor (batch_size, seq_len)
        :return mean: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :return std: torch.FloatTensor (num_layers, batch_size, hidden_size)
        '''

        final_states = self.encoder(src)
        mean = self.mean_projection(final_states)
        # std = F.softplus(self.std_projection(final_states))
        std = torch.exp(self.std_projection(final_states))
        return mean, std

    def reparametrize(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        '''
        :param mean: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :param std: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :return encoding: torch.FloatTensor (num_layers, batch_size, hidden_size)
        '''

        noise = torch.randn(size=mean.size(), device=mean.device)
        encoding = mean + std * noise
        return encoding

    def probabilistic_encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        :param src: torch.LongTensor (batch_size, seq_len)
        :return encoding: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :return mean: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :return std: torch.FloatTensor (num_layers, batch_size, hidden_size)
        '''

        mean, std = self.encode(src)
        encoding = self.reparametrize(mean, std)
        return encoding, mean, std

    def _word_dropout(self, trg: torch.Tensor) -> torch.Tensor:
        '''
        :param trg: torch.LongTensor (batch_size, seq_len)
        :return mask * trg: torch.LongTensor (batch_size, seq_len)
        '''

        mask = (trg == PAD_INDEX)
        p = torch.FloatTensor(mask.size()).to(trg.device)
        constant_(p, 1 - self.word_dropout)
        p.masked_fill_(mask, 1)
        mask = torch.bernoulli(p).long()
        return mask * trg

    def sample(self, **kwargs):
        '''
        :num: int
        :encoding: torch.FloatTensor (num_layers, sample_num, hidden_size)
        :output: torch.LongTensor (sample_num, seq_len)
        :logit: torch.FloatTensor (sample_num, seq_len, vocab_size)
        '''

        assert ('num' in kwargs) ^ ('encoding' in kwargs)
        if 'num' in kwargs:
            encoding = torch.randn(size=(self.num_layers, kwargs['num'], self.hidden_size),
                                   device=self.encoder.embedding.weight.device)
        else:
            encoding = kwargs['encoding']
        max_len = kwargs.get('max_len', 15)
        logit = self.decoder.decode(encoding, max_len)
        output = logit.argmax(dim=-1)
        return_list = [output]
        if kwargs.get('output_encoding', False):
            num = encoding.size(1)
            encoding = encoding.transpose(0, 1).reshape(num, -1).cpu().numpy()
            return_list.append(encoding)
        if kwargs.get('output_logit', False):
            return_list.append(logit)
        return return_list[0] if len(return_list) == 1 else return_list