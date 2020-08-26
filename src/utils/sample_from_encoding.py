import torch
from torch import nn
from torchtext.vocab import Vocab
from typing import List
from src.utils.convert_tensor_to_texts import convert_tensor_to_texts

def sample_from_encoding(model: nn.Module, vocab: Vocab, encoding: torch.Tensor, max_len: int, batch_size: int = 64) -> List[str]:
    '''
    :param model: vae nn.Module
    :param vocab: Vocab
    :param encoding: torch.FloatTensor (num_layers, sample_num, hidden_size)
    :param batch_size: int
    :return sentences: List[str]
    '''

    sample_num = encoding.size(1)
    sentences = []
    start = 0
    model.eval()

    with torch.no_grad():
        while start < sample_num:
            end = min(sample_num, start + batch_size)
            output = model.sample(encoding=encoding[:, start:end, :], max_len=max_len)
            sentences.extend(convert_tensor_to_texts(output, vocab))
            start = end
    return sentences