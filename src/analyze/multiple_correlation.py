import torch

def multiple_correlation(X: torch.Tensor, y: torch.Tensor) -> float:
    """
    :param X: torch.FloatTensor (num, dim)
    :param y: torch.FloatTensor (num,)
    """
    assert len(X.size()) == 2
    assert len(y.size()) == 1
    assert X.size(0) == y.size(0)
    num = X.size(0)
    ones = torch.ones(size=(num, 1), dtype=X.dtype, device=X.device)
    X = torch.cat((X, ones), dim=1)
    Y = y.unsqueeze(-1)
    W = torch.inverse(X.t().matmul(X)).matmul(X.t()).matmul(Y)  # torch.FloatTensor (dim + 1, 1)
    Z = X.matmul(W)
    z = Z.squeeze(-1)
    y = y - y.mean()
    z = z - z.mean()
    correlation = y.dot(z) / (torch.norm(y) * torch.norm(z))
    return correlation.item()

def get_linear_weights(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    :param X: torch.FloatTensor (num, dim)
    :param y: torch.FloatTensor (num,)
    """
    assert len(X.size()) == 2
    assert len(y.size()) == 1
    assert X.size(0) == y.size(0)
    num = X.size(0)
    ones = torch.ones(size=(num, 1), dtype=X.dtype, device=X.device)
    X = torch.cat((X, ones), dim=1)
    Y = y.unsqueeze(-1)
    W = torch.inverse(X.t().matmul(X)).matmul(X.t()).matmul(Y)  # torch.FloatTensor (dim + 1, 1)
    w = W[:-1, 0]
    return w