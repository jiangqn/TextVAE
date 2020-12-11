import torch
from torch import nn

class InvertibleMLP(nn.Module):

    def __init__(self, hidden_size, target_size):
        super(InvertibleMLP, self).__init__()
        self.transform = nn.Sequential(
            InvertibleLinear(hidden_size),
            nn.LeakyReLU(),
            InvertibleLinear(hidden_size),
            nn.LeakyReLU()
        )
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        return self.linear(self.transform(X))

class InvertibleLinear(nn.Module):

    def __init__(self, hidden_size):
        super(InvertibleLinear, self).__init__()
        self.x = nn.Parameter(torch.randn(hidden_size, 1))
        self.y = nn.Parameter(torch.randn(hidden_size, 1))
        self.bias = nn.Parameter(torch.randn(1, hidden_size))
        self.hidden_size = hidden_size

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        dot = self.x.t().matmul(self.y)
        weight = torch.eye(self.hidden_size, device=X.device) + self.x.matmul(self.y.t()) * ((torch.exp(dot) - 1) / dot)
        return X.matmul(weight)# + self.bias