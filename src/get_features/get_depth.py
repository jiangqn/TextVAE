import pandas as pd
from typing import List
import multiprocessing
from src.utils.sentence_depth import sentence_depth

def get_depth(sentences: List[str], **kwargs) -> List[int]:
    processes = kwargs.get('processes', 1)
    if processes > 1:
        pool = multiprocessing.Pool(processes=processes)
        return pool.map(sentence_depth, sentences)
    else:
        return list(map(sentence_depth, sentences))

def get_depth_from_tsv(path: str, **kwargs) -> List[int]:
    df = pd.read_csv(path, delimiter='\t')
    sentences = list(df['sentence'])
    return get_depth(sentences, **kwargs)