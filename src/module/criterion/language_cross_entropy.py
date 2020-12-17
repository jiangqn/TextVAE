import torch
from torch import nn
from typing import Tuple

class LanguageCrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index: int = None):
        super(LanguageCrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.ignore_index = ignore_index


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param input: torch.FloatTensor (batch_size, seq_len, vocab_size)
        :param target: torch.LongTensor (batch_size, seq_len)
        :return nll: torch.FloatTensor (batch_size,)
        :return seq_lens: torch.FloatTensor (batch_size,)
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
            seq_lens = mask.float().sum(dim=1)
        else:
            seq_lens = torch.tensor([seq_len for _ in range(batch_size)]).float().to(input.device)

        nll = loss.sum(dim=1)
        return nll, seq_lens