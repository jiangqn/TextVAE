import torch
from torchtext.vocab import Vocab
from typing import List
from src.constants import EOS_INDEX

def convert_tensor_to_texts(tensor: torch.LongTensor, vocab: Vocab) -> List[str]:
    f = lambda line: ' '.join([vocab.itos[index] for index in line])
    indices = tensor.tolist()
    texts = []
    for line in indices:
        if EOS_INDEX in line:
            eos_position = line.index(EOS_INDEX)
            line = line[0: eos_position]
        texts.append(f(line))
    return texts