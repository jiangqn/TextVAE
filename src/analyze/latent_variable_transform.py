import torch
from src.analyze.iresnet import InvertibleResNet

def minimal_norm_solve(latent_variable: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    :param latent_variable: torch.FloatTensor (num, dim)
    :param target: torch.FloatTensor (num,)
    :param weight: torch.FloatTensor (dim,)
    :param weight: torch.FloatTensor (1,)
    """
    prediction = latent_variable.matmul(weight) + bias
    weight = weight.unsqueeze(0)
    target = target.unsqueeze(1)
    prediction = prediction.unsqueeze(1)
    return (target - prediction).matmul(1 / weight.matmul(weight.t())).matmul(weight) + latent_variable

def transform(model: InvertibleResNet, latent_variable: torch.Tensor, batch_size: int = 100) -> torch.Tensor:
    num = latent_variable.size(0)
    transformed_latent_variable = []

    start = 0
    model.eval()

    with torch.no_grad():
        while start < num:
            end = min(num, start + batch_size)
            transformed_latent_variable.append(model.transform(latent_variable[start:end, :]))
            start = end

    transformed_latent_variable = torch.cat(transformed_latent_variable, dim=0)
    return transformed_latent_variable

def inverse_transform(model: InvertibleResNet, transformed_latent_variable: torch.Tensor, batch_size: int = 100) -> torch.Tensor:
    num = transformed_latent_variable.size(0)
    latent_variable = []

    start = 0
    model.eval()

    with torch.no_grad():
        while start < num:
            end = min(num, start + batch_size)
            latent_variable.append(model.inverse_transform(transformed_latent_variable[start:end, :]))
            start = end

    latent_variable = torch.cat(latent_variable, dim=0)
    return latent_variable