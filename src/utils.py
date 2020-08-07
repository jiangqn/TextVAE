import os
import numpy as np
from src.constants import PAD_INDEX, EOS_INDEX

def sentence_clip(sentence):
    mask = (sentence != PAD_INDEX)
    sentence_lens = mask.long().sum(dim=1, keepdim=False)
    max_len = sentence_lens.max().item()
    return sentence[:, :max_len]

def convert_tensor_to_texts(tensor, vocab):
    f = lambda line: ' '.join([vocab.itos[index] for index in line])
    indices = tensor.tolist()
    texts = []
    for line in indices:
        if EOS_INDEX in line:
            eos_position = line.index(EOS_INDEX)
            line = line[0: eos_position]
        texts.append(f(line))
    return texts

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