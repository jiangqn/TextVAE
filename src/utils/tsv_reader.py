import pandas as pd
from src.get_features.get_length import get_length
from src.get_features.get_depth import get_depth

def read_field(path: str, field: str) -> list:
    df = pd.read_csv(path, delimiter='\t')
    assert field in df.columns.values.tolist()
    return list(df[field])

def read_prop(path, prop, **kwargs):
    assert prop in ['length', 'depth']
    if prop == 'length':
        return read_sentence_length(path)
    else:
        return read_sentence_depth(path, **kwargs)

def read_sentence_length(path):
    df = pd.read_csv(path, delimiter='\t')
    sentences = list(df['sentence'])
    length = get_length(sentences)
    return length

def read_sentence_depth(path, **kwargs):
    df = pd.read_csv(path, delimiter='\t')
    sentences = list(df['sentence'])
    depth = get_depth(sentences, **kwargs)
    return depth