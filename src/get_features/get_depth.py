import pandas as pd
from typing import List
import multiprocessing
from src.utils.sentence_depth import sentence_depth
from supar import Parser

def get_depth(sentences: List[str], **kwargs) -> List[int]:

    def compute_depth(sentence):
        depth = 0
        count = 0
        for char in sentence:
            if char == "(":
                count += 1
            elif char == ")":
                count -= 1
            depth = max(depth, count)
        return depth

    input_dataset = [sentence.replace("(", "[").replace(")", "]").split() for sentence in sentences]
    n = len(sentences)
    depths = []
    remain_ids = []
    for i in range(n):
        if len(input_dataset[i]) == 1:
            depths.append(1)
        else:
            depths.append(0)
            remain_ids.append(i)

    parser = Parser.load("crf-con-en")
    dataset = parser.predict(input_dataset, prob=True, verbose=False)

    n_remain = len(dataset.sentences)
    remain_depths = []
    for i in range(n_remain):
        remain_depths.append(compute_depth(str(dataset.sentences[i])))

    for id, depth in zip(remain_ids, remain_depths):
        depths[id] = depth

    return depths

# def get_depth(sentences: List[str], **kwargs) -> List[int]:
#     processes = kwargs.get('processes', 1)
#     if processes > 1:
#         pool = multiprocessing.Pool(processes=processes)
#         return pool.map(sentence_depth, sentences)
#     else:
#         return list(map(sentence_depth, sentences))

def get_depth_from_tsv(path: str, **kwargs) -> List[int]:
    df = pd.read_csv(path, delimiter='\t')
    sentences = list(df['sentence'])
    return get_depth(sentences, **kwargs)