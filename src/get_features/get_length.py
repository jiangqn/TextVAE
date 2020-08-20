import pandas as pd
from typing import List

def get_length(sentences: List[str]) -> List[int]:
    return list(map(lambda sentence: len(sentence.split()), sentences))

def get_length_from_tsv(path: str) -> List[int]:
    df = pd.read_csv(path, delimiter='\t')
    sentences = list(df['sentence'])
    return get_length(sentences)