import pandas as pd
from typing import List
from src.utils.sentence_depth import sentence_depth

def get_depth(sentences: List[str]) -> List[int]:
    return list(map(sentence_depth, sentences))

def get_depth_from_tsv(path: str) -> List[int]:
    df = pd.read_csv(path, delimiter='\t')
    sentences = list(df['sentence'])
    return get_depth(sentences)