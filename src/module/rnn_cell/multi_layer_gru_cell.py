import torch
from torch import nn
import torch.nn.functional as F

class MultiLayerGRUCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0, bias: bool = True) -> None:
        super(MultiLayerGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias
        self.gru_cells = nn.ModuleList(
            [nn.GRUCell(input_size, hidden_size, bias)] + [nn.GRUCell(hidden_size, hidden_size, bias) for _ in range(num_layers - 1)]
        )

    def forward(self, input: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        :param input: torch.FloatTensor (batch_size, seq_len, input_size)
        :param states: torch.FloatTensor (num_layers, batch_size, hidden_size)
        :return output_hidden: torch.FloatTensor (num_layers, batch_size, hidden_size)
        """
        hidden = states
        output_hidden = []
        for i, gru_cell in enumerate(self.gru_cells):
            h = gru_cell(input, hidden[i])
            output_hidden.append(h)
            input = F.dropout(h, p=self.dropout, training=self.training) # ??? in last layer
        output_hidden = torch.stack(output_hidden, dim=0)
        return output_hidden