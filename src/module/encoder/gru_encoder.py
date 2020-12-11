import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.module.encoder.encoder import Encoder
from src.constants import PAD_INDEX

class GRUEncoder(Encoder):

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super(GRUEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = dropout

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        '''
        encode sentences into vector representations

        :param src: torch.LongTensor (batch_size, seq_len)
        :return final_states: torch.FloatTensor (num_layers, batch_size, hidden_size * 2)
        '''

        src_mask = (src != PAD_INDEX)
        src_lens = src_mask.long().sum(dim=1, keepdim=False)

        src_embedding = self.embedding(src)
        src = F.dropout(src_embedding, p=self.dropout, training=self.training)

        src_lens, sort_index = src_lens.sort(descending=True)
        src = src.index_select(dim=0, index=sort_index)
        packed_src = pack_padded_sequence(src, src_lens, batch_first=True)

        packed_output, final_states = self.rnn(packed_src)

        reorder_index = sort_index.argsort(descending=False)
        final_states = final_states.index_select(dim=1, index=reorder_index)

        final_states = torch.cat(final_states.chunk(chunks=2, dim=0), dim=2)
        return final_states