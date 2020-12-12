import torch
from torch import nn
import torch.nn.functional as F
from src.module.encoder.encoder import Encoder
from src.constants import PAD_INDEX
from typing import List

class ConvEncoder(Encoder):

    """
    The convolutional encoder of TextVAE
    """

    def __init__(self, vocab_size: int, embed_size: int, kernel_sizes: List[int], kernel_num: int, num_layers: int, dropout: float) -> None:
        super(ConvEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.feature_size = len(kernel_sizes) * kernel_num
        for i in range(num_layers):
            self.conv_layers.append(ConvEncoderLayer(
                input_size=embed_size if i == 0 else self.feature_size,
                kernel_sizes=kernel_sizes,
                kernel_num=kernel_num,
                dropout=dropout
            ))
            self.layer_norms.append(nn.LayerNorm(self.feature_size))
        self.dropout = dropout

    @property
    def output_size(self) -> int:
        return self.feature_size

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: torch.LongTensor (batch_size, seq_len)
        representation: torch.FloatTensor (batch_size, hidden_size)
        """
        mask = (src != PAD_INDEX).float()
        src_len = mask.sum(dim=1, keepdim=True)
        embedding = self.embedding(src)
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        activation = embedding
        for conv_layer, layer_norm in zip(self.conv_layers, self.layer_norms):
            activation = conv_layer(activation)
            activation = layer_norm(activation)
        representation = torch.mean(activation, dim=-1)
        return representation

class ConvEncoderLayer(nn.Module):

    def __init__(self, input_size: int, kernel_sizes: List[int], kernel_num: int, dropout: float) -> None:
        super(ConvEncoderLayer, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.kernel_num = kernel_num
        self.convs = nn.ModuleList(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=kernel_num,
                kernel_size=kernel_size
            ) for kernel_size in kernel_sizes
        )
        self.dropout = dropout

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: torch.FloatTensor (batch_size, input_size, seq_len)
        output: torch.FloatTensor (batch_size, output_size, seq_len)
        output_size = len(kernel_sizes) * kernel_num
        """
        conv_output = []
        for kernel_size, conv in zip(self.kernel_sizes, self.convs):
            conv_output.append(conv(F.pad(input, pad=[(kernel_size - 1) // 2, kernel_size // 2, 0, 0, 0, 0])))
        output = torch.cat(conv_output, dim=1)
        output = F.gelu(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output