import math
import numpy as np
from typing import List

def mean(X: list) -> float:
    return sum(X) / len(X)

def accuracy(X: List[int], Y: List[int]) -> float:
    assert len(X) == len(Y)
    return mean([int(x == y) for x, y in zip(X, Y)])

def rmse(X: list, Y: list) -> float:
    return math.sqrt(mean([(x - y) ** 2 for x, y in zip(X, Y)]))

def MAE(X: list, Y: list) -> float:
    return mean([abs(x - y) for x, y in zip(X, Y)])

def correlation(X: list, Y: list) -> float:
    X = np.asarray(X).astype(np.float32)
    Y = np.asarray(Y).astype(np.float32)
    return float(np.corrcoef(X, Y)[0, 1])