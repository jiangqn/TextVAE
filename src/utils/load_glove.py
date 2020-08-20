import os
import numpy as np
from src.constants import PAD_INDEX

def load_glove(path, vocab_size, word2index):
    if not os.path.isfile(path):
        raise IOError('Not a file', path)
    glove = np.random.uniform(-0.01, 0.01, [vocab_size, 300])
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            if content[0] in word2index:
                glove[word2index[content[0]]] = np.array(list(map(float, content[1:])))
    glove[PAD_INDEX, :] = 0
    return glove