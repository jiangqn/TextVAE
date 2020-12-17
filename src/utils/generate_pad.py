import torch
from torch.nn import init
from src.constants import PAD_INDEX
from typing import Union, List, Tuple

def generate_pad(size: Union[torch.Size, List[int], Tuple[int]], device: torch.device = torch.device("cpu")) -> torch.Tensor:
    pad = torch.LongTensor(size=size).to(device)
    init.constant_(pad, PAD_INDEX)
    return pad