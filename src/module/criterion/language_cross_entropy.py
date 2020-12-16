import torch
from torch import nn

class LanguageCrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index: int = None, batch_reduction: str = "mean", seq_reduction: str = "sum"):
        super(LanguageCrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.ignore_index = ignore_index
        assert batch_reduction in ["none", "mean", "sum"]
        self.batch_reduction = batch_reduction
        assert seq_reduction in ["mean", "sum"]
        self.seq_reduction = seq_reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param input: torch.FloatTensor (batch_size, seq_len, vocab_size)
        :param target: torch.LongTensor (batch_size, seq_len)
        :param loss: torch.FloatTensor (1,) or (batch_size,)
        """
        assert len(input.size()) == 3
        assert len(target.size()) == 2
        assert input.size(0) == target.size(0) and input.size(1) == target.size(1)

        batch_size, seq_len, vocab_size = input.size()
        input = input.view(batch_size * seq_len, vocab_size)
        target = target.view(batch_size * seq_len)
        loss = self.cross_entropy(input, target)    # torch.FloatTensor (batch_size * seq_len)
        loss = loss.view(batch_size, seq_len)
        target = target.view(batch_size, seq_len)

        if self.ignore_index != None:
            mask = (target != self.ignore_index)
            loss.masked_fill_(mask==0, 0)
            batch_seq_len = mask.float().sum(dim=1)

            if self.seq_reduction == "sum":
                loss = loss.sum(dim=1)
            else:   # self.seq_reduction == "mean"
                loss = loss.sum(dim=1) / batch_seq_len

        else:   # self.ignore_index == None

            if self.seq_reduction == "sum":
                loss = loss.sum(dim=1)
            else:   # self.seq_reduction == "mean"
                loss = loss.mean(dim=1)

        if self.batch_reduction == "sum":
            loss = loss.sum()
        elif self.batch_reduction == "mean":
            loss = loss.mean()
        # else self.batch_reduction == "none"

        return loss