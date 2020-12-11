import torch
from torch import nn

class Decoder(nn.Module):

    """
    The base class of decoder of TextVAE
    """

    def __init__(self) -> None:
        super(Decoder, self).__init__()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The forward method in class Encoder is not implemented.")