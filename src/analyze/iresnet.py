import torch
from torch import nn
import torch.nn.functional as F
from src.analyze.spectral_norm import SpectralNorm
# torch.set_default_dtype(torch.float64)

class InvertibleResNet(nn.Module):

    def __init__(self, hidden_size: int, n_blocks: int = 1, output_size: int = 1) -> None:
        super(InvertibleResNet, self).__init__()
        self.n_blocks = n_blocks
        self.residual_blocks = nn.ModuleList(InvertibleResidualBlock(hidden_size) for _ in range(n_blocks))
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        Y = self.transform(X)
        return self.linear(Y)

    def transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        for i in range(self.n_blocks):
            X = self.residual_blocks[i](X)
        return X

    def inverse_transform(self, Y: torch.FloatTensor) -> torch.FloatTensor:
        with torch.no_grad():
            for i in range(self.n_blocks - 1, -1, -1):
                Y = self.residual_blocks[i].inverse(Y)
            return Y

    def check(self):
        for i in range(self.n_blocks):
            self.residual_blocks[i].check()

class InvertibleResidualBlock(nn.Module):

    def __init__(self, hidden_size: int) -> None:
        super(InvertibleResidualBlock, self).__init__()
        self.linear1 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False
        )
        self.linear2 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False
        )
        self.transform = nn.Sequential(
            SpectralNorm(self.linear1),
            nn.ReLU(),
            SpectralNorm(self.linear2)
        )
        self.negative_slope=0.01

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        return F.leaky_relu(self.transform(X) + X, negative_slope=self.negative_slope)

    def inverse(self, Y: torch.FloatTensor) -> torch.FloatTensor:
        with torch.no_grad():

            Y = Y.clone()
            Y = F.leaky_relu(Y, negative_slope=1 / self.negative_slope)

            Z = Y.clone()
            for i in range(20):
                Z = Y - self.linear2(torch.relu(self.linear1(Z)))
            return Z

    def check(self):
        W1 = self.linear1.weight
        _, s, _ = torch.svd(W1)
        print(s.max().item())
        W2 = self.linear1.weight
        _, s, _ = torch.svd(W2)
        print(s.max().item())