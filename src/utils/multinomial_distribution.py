import torch
from collections import Counter
from typing import List
from src.utils.tsv_reader import read_field

def get_multinomial_distribution(data: List[int]) -> List[float]:
    num = len(data)
    max_item = max(data)
    counter = Counter(data)
    distribution = [(counter[i] if i in counter else 0) / num for i in range(max_item + 1)]
    return distribution

def get_multinomial_distribution_from_tsv(path: str, field: str) -> List[float]:
    data = read_field(path, field)
    return get_multinomial_distribution(data)

def sample_from_multinomial_distribution(distribution: List[float], sample_num: int) -> List[int]:
    distribution = torch.tensor(distribution)
    sample = torch.multinomial(distribution, sample_num, replacement=True).tolist()
    return sample