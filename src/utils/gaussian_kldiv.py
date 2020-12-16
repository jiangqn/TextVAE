import torch
from torch import nn

class GaussianKLDiv(nn.Module):

    def __init__(self, reduction: str = "mean") -> None:
        super(GaussianKLDiv, self).__init__()
        assert reduction in [None, "mean", "sum"]
        self.reduction = reduction

    def forward(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        :param mean: torch.FloatTensor (batch_size, latent_size)
        :param std: torch.FloatTensor (batch_size, latent_size)
        """
        assert mean.size() == std.size()
        kld = 0.5 * ( -torch.log(std * std) + mean * mean + std * std - 1).sum(dim=-1)
        if self.reduction == "mean":
            kld = kld.mean()
        elif self.reduction == "sum":
            kld = kld.sum()
        # else self.reduction == "none"
        return kld