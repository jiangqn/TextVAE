import torch
from torch import nn

class Encoder(nn.Module):

    """
    The base class of encoder of TextVAE
    """

    def __init__(self) -> None:
        super(Encoder, self).__init__()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The forward method in class Encoder is not implemented.")

    @property
    def output_size(self) -> int:
        raise NotImplementedError("The @property output_size method in class Encoder is not implemented.")