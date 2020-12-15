import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class MultiLayerLSTMCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, latent_size: int, num_layers: int = 1, dropout: float = 0, bias: bool = True) -> None:
        super(MultiLayerLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias
        self.lstm_cells = nn.ModuleList(
            [nn.LSTMCell(input_size + latent_size, hidden_size, bias)] + [nn.LSTMCell(hidden_size + latent_size, hidden_size, bias) for _ in range(num_layers - 1)]
        )

    def forward(self, input: torch.Tensor, states: torch.Tensor, latent_variable: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param input: FloatTensor (batch_size, input_size)
        :param states: tuple (hidden, cell)
            hidden: FloatTensor (num_layers, batch_size, hidden_size)
            cell: FloatTensor (num_layers, batch_size, hidden_size)
        :param latent_variable: FloatTensor (batch_size, latent_size)
        :return hidden: FloatTensor (num_layers, batch_size, hidden_size)
        :return cell: FloatTensor (num_layers, batch_size, hidden_size)
        """
        hidden, cell = states
        output_hidden = []
        output_cell = []
        for i, lstm_cell in enumerate(self.lstm_cells):
            h, c = lstm_cell(torch.cat((input, latent_variable), dim=-1), (hidden[i], cell[i]))
            output_hidden.append(h)
            output_cell.append(c)
            input = F.dropout(h, p=self.dropout, training=self.training)
        output_hidden = torch.stack(output_hidden, dim=0)
        output_cell = torch.stack(output_cell, dim=0)
        return output_hidden, output_cell