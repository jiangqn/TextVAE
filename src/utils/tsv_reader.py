import pandas as pd
from src.utils.sentence_depth import sentence_depth

def read_prop(path, prop):
    assert prop in ['length', 'depth']
    if prop == 'length':
        return read_sentence_length(path)
    else:
        return read_sentence_depth(path)

def read_sentence_length(path):
    df = pd.read_csv(path, delimiter='\t')
    sentences = list(df['sentence'])
    length = [len(sentence.split()) for sentence in sentences]
    return length

def read_sentence_depth(path):
    df = pd.read_csv(path, delimiter='\t')
    sentences = list(df['sentence'])
    depth = [sentence_depth(sentence) for sentence in sentences]
    return depth