import torch

def minimal_norm_solve(latent_variable: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    :param latent_variable: torch.FloatTensor (num, dim)
    :param target: torch.FloatTensor (num,)
    :param weight: torch.FloatTensor (dim,)
    """
    prediction = latent_variable.matmul(weight)
    weight = weight.unsqueeze(0)
    target = target.unsqueeze(1)
    prediction = prediction.unsqueeze(1)
    return (target - prediction).matmul(1 / weight.matmul(weight.t())).matmul(weight) + latent_variable