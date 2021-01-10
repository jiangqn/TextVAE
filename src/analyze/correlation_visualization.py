import torch
#from src.analyze.constraint_max_variance_direction_solver import solve

def correlation_visualization(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    :param X: torch.FloatTensor (num, dim)
    :param y: torch.FloatTensor (num,)
    """
    assert len(X.size()) == 2
    assert len(y.size()) == 1
    assert X.size(0) == y.size(0)

    X = X - X.mean(dim=0, keepdim=True)

    num, dim = X.size()
    ones = torch.ones(size=(num, 1), dtype=X.dtype, device=X.device)
    X = torch.cat((X, ones), dim=1)
    Y = y.unsqueeze(-1)

    U = torch.inverse(X.t().matmul(X)).matmul(X.t()).matmul(Y)

    X = X[:, 0:-1]
    U = U[0:-1, :]
    U = U / U.norm(dim=0, keepdim=True)

    T = torch.randn(dim, dim, dtype=X.dtype, device=X.device)
    T[:, 0] = U[:, 0]
    T, _ = torch.qr(T)
    T[:, 0] = U[:, 0]

    tX = X.matmul(T)
    tX = tX[:, 1:]
    cov = tX.t().matmul(tX)

    eigenvalues, eigenvectors = torch.eig(cov, eigenvectors=True)
    eigenvalues = eigenvalues[:, 0]
    _, index = eigenvalues.max(dim=0)
    eigenvector = eigenvectors[index]

    zero = torch.zeros(1, dtype=X.dtype, device=X.device)
    v = torch.cat([zero, eigenvector], dim=0)
    V = v.unsqueeze(-1)
    V = T.matmul(V)
    V = V / V.norm(dim=0, keepdim=True)

    P = torch.cat([U, V], dim=-1)
    Z = X.matmul(P)
    
    return Z
