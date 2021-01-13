import torch

def has_nan(X: torch.Tensor) -> bool:
    return torch.any(torch.isnan(X)).item()