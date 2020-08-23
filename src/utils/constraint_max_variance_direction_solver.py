import torch
from torch import nn
from torch import optim
from copy import deepcopy

class ConstraintMaxVarianceDirectionSolver(nn.Module):
    '''
    Find the direction u that maximizes the variance of projection of data matrix X with the following 2 constraints
    by gradient-based optimization.

    Constraint 1: norm(u) = 1
    Constraint 2: u.dot(v) = 0, where norm(v) = 1
    '''


    def __init__(self, dim: int, lambd: int = -1, mu: int = -1) -> None:
        super(ConstraintMaxVarianceDirectionSolver, self).__init__()
        self.u = nn.Parameter(torch.rand(dim, 1))
        self.lambd = lambd
        self.mu = mu

    def forward(self, X: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return - (self.u.t().matmul(X).matmul(self.u)[0, 0] + self.lambd * torch.abs(self.u.t().matmul(self.u) - 1) + self.mu * torch.abs(self.u.t().matmul(v)))

def solve(X: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    '''
    :param X: torch.FloatTensor (num, encoding_size)
    :param v: torch.FloatTensor (encoding_size,)
    :return u: torch.FloatTensor (encoding_size,)
    '''

    dim = X.size(1)

    solver = ConstraintMaxVarianceDirectionSolver(dim, -10, -10).cuda()
    optimizer = optim.SGD(solver.parameters(), lr=0.0003, momentum=0.9)

    X = X.cuda()
    v = v.cuda()

    n_step = 10000

    min_loss = 1e9
    solution = None

    for i in range(n_step):
        optimizer.zero_grad()
        loss = solver(X, v)
        loss.backward()
        if loss.item() < min_loss:
            min_loss = loss.item()
            solution = deepcopy(solver.u.data)
        optimizer.step()

    return solution