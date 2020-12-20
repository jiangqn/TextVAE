import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.module.encoder.encoder import Encoder
from src.constants import PAD_INDEX
from src.module.attention import get_attention

class LSTMEncoder(Encoder):

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, bidirectional: bool,
                 dropout: float, output_type: str) -> None:
        super(LSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = dropout
        self.num_directions = 2 if self.bidirectional else 1
        self.feature_size = self.hidden_size * self.num_directions
        assert output_type in ["final_hidden", "final_cell", "average_pooling", "max_pooling", "attention_pooling"]
        self.output_type = output_type
        if output_type == "attention_pooling":
            self.attention = get_attention(
                query_size=self.feature_size,
                key_size=self.feature_size,
                attention_type="Bilinear"
            )

    @property
    def output_size(self) -> int:
        return self.feature_size

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        encode sentences into vector representations

        :param src: torch.LongTensor (batch_size, seq_len)
        :return encoder_output: torch.FloatTensor (batch_size, output_size)
        """

        batch_size = src.size(0)

        src_mask = (src != PAD_INDEX)
        src_len = src_mask.long().sum(dim=1, keepdim=False) # torch.LongTensor (batch_size,)

        src_embedding = self.embedding(src)
        src = F.dropout(src_embedding, p=self.dropout, training=self.training)  # torch.FloatTensor (batch_size, seq_len, embed_size)

        # sort and pack
        src_len, sort_index = src_len.sort(descending=True)
        src = src.index_select(dim=0, index=sort_index)
        packed_src = pack_padded_sequence(src, src_len, batch_first=True)

        # encode
        packed_output, (final_hidden, final_cell) = self.rnn(packed_src)

        # unpack (pad) and reorder
        output = pad_packed_sequence(packed_output, batch_first=True)[0]   # torch.FloatTensor (batch_size, seq_len, output_size)
        reorder_index = sort_index.argsort(descending=False)
        final_hidden = final_hidden.index_select(dim=1, index=reorder_index)    # torch.FloatTensor (num_layers * num_directions, batch_size, hidden_size)
        final_cell = final_cell.index_select(dim=1, index=reorder_index)
        output = output.index_select(dim=0, index=reorder_index)    # torch.FloatTensor (batch_size, seq_len, num_directions * hidden_size)

        #final_hidden = torch.cat(final_hidden.chunk(chunks=2, dim=0), dim=2)

        final_hidden = final_hidden.reshape(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        final_hidden = torch.cat([final_hidden[:, i, :, :] for i in range(self.num_directions)], dim=-1)[0]
        final_cell = final_cell.reshape(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        final_cell = torch.cat([final_cell[:, i, :, :] for i in range(self.num_directions)], dim=-1)[0]

        if self.output_type == "final_hidden":
            encoder_output = final_hidden
        elif self.output_type == "final_cell":
            encoder_output = final_cell
        elif self.output_type == "average_pooling":
            encoder_output = (output * src_mask.unsqueeze(-1).float()).sum(dim=1, keepdim=False) / src_len.unsqueeze(-1).float()
        elif self.output_type == "max_pooling":
            encoder_output = output.max(dim=1)[0]
        else:   # self.output_type == "attention"
            encoder_output = self.attention(final_hidden, output, output, src_mask)

        return encoder_output