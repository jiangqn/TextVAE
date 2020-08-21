import os
import torch
from torch import nn
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import numpy as np
import pickle
from src.constants import EOS

def sentence_encoding(model: nn.Module, data_iter: Iterator, probabilistic_encoding: bool = False) -> torch.Tensor:
    '''
    :param model: vae nn.Module
    :param data_iter:
    :param probabilistic_encoding:
    :return encoding: torch.FloatTensor (num_layers, num, hidden_size)
    '''

    encoding = []

    model.eval()

    with torch.no_grad():

        for batch in data_iter:

            sentence = batch.sentence
            if probabilistic_encoding:
                batch_encoding, _, _ = model.probabilistic_encode(sentence)
            else:
                batch_encoding, _ = model.encode(sentence)

            # batch_encoding = batch_encoding.cpu().numpy()
            encoding.append(batch_encoding)

    encoding = torch.cat(encoding, dim=1)
    return encoding

def sentence_encoding_from_tsv(file_path: str, **kwargs) -> torch.Tensor:

    base_path = os.path.dirname(file_path)
    vocab_path = os.path.join(base_path, 'vocab.pkl')

    assert ('model_path' in kwargs) ^ ('model' in kwargs)

    if 'model_path' in kwargs:
        model = torch.load(kwargs['model_path'])
    else:
        model = torch.load(kwargs['model'])

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, eos_token=EOS)
    fields = [('sentence', TEXT)]

    dataset = TabularDataset(file_path, format='tsv', skip_header=True, fields=fields)
    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)
    TEXT.vocab = vocab

    device = model.encoder.embedding.weight.device
    batch_size = kwargs.get('batch_size', 64)
    data_iter = Iterator(dataset, batch_size=batch_size, shuffle=False, device=device)

    probabilistic_encoding = kwargs.get('probabilistic_encoding', False)

    return sentence_encoding(model, data_iter, probabilistic_encoding=probabilistic_encoding)