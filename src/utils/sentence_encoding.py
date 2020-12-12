import os
import torch
from torch import nn
from torch import optim
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import numpy as np
import math
import pickle
from src.constants import SOS, EOS, PAD_INDEX
from src.utils.gaussian_kldiv import GaussianKLDiv

def gradient_encoding(model: nn.Module, sentence: torch.Tensor, **kwargs) -> torch.Tensor:

    lr = kwargs.get('lr', 0.01)
    weight_deacy = kwargs.get('weight_decay', 0)
    max_iter = kwargs.get('max_iter', 500)

    src = sentence[:, 1:]
    trg_input = sentence
    batch_size = sentence.size(0)
    pad = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=sentence.device)
    trg_output = torch.cat((sentence[:, 1:], pad), dim=-1)

    encoding, _ = model.encode(src)
    encoding = nn.Parameter(encoding)

    optimizer = optim.Adam([encoding], lr=lr, weight_decay=weight_deacy)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
    kldiv =GaussianKLDiv()

    min_loss = 1e9
    best_encoding = encoding.data
    best_i = 0

    for i in range(max_iter):

        optimizer.zero_grad()

        logit = model.decoder(encoding, trg_input)
        trg_output = trg_output.view(-1)
        output_size = logit.size(-1)
        logit = logit.view(-1, output_size)

        std = torch.zeros_like(encoding, device=encoding.device)
        nn.init.constant_(std, 1)
        norm = (encoding * encoding).sum(dim=0).sum(dim=-1).sqrt().mean()

        rec_loss = criterion(logit, trg_output)
        kl_loss = kldiv(encoding, std)
        loss = rec_loss + 0.04 * kl_loss
        loss.backward()

        optimizer.step()

        logprob = - ((math.log(2 * math.pi) + encoding * encoding) / 2).sum(dim=0).sum(dim=-1).mean().item()
        print('%d\t%.4f\t%.4f\t%.4f\t%.4f' % (i, rec_loss.item(), kl_loss.item(), norm.item(), logprob))

        if loss.item() < min_loss:
            min_loss = loss.item()
            best_encoding = encoding.data
            best_i = i

    print(best_i)

    # raise ValueError('debug')

    return best_encoding

def sentence_encoding(model: nn.Module, sentence: torch.Tensor, **kwargs) -> torch.Tensor:
    '''
    :param model: vae nn.Module
    :param sentence: torch.LongTensor (batch_size, seq_len)
    :param kwargs:
    :return encoding: torch.FloatTensor (num_layers, batch_size, hidden_size)
    '''

    encoding_type = kwargs.get('encoding_type', 'deterministic')
    assert encoding_type in ['deterministic', 'probabilistic', 'gradient']

    src = sentence[:, 1:]

    if encoding_type == 'deterministic':
        encoding, _ = model.encode(src)
    elif encoding_type == 'probabilistic':
        encoding, _, _ = model.probabilistic_encode(src)
    else:   # gradient
        encoding = gradient_encoding(model, sentence, **kwargs)
    return encoding

def sentence_encoding_from_iterator(model: nn.Module, data_iter: Iterator, **kwargs) -> torch.Tensor:
    '''
    :param model: vae nn.Module
    :param data_iter:
    :param probabilistic_encoding:
    :return encoding: torch.FloatTensor (num_layers, num, hidden_size)
    '''

    encoding = []

    model.eval()

    for batch in data_iter:
        encoding.append(sentence_encoding(model, batch.sentence, **kwargs))

    encoding = torch.cat(encoding, dim=1)
    return encoding

def sentence_encoding_from_tsv(file_path: str, **kwargs) -> torch.Tensor:

    base_path = os.path.dirname(file_path)
    vocab_path = os.path.join(base_path, 'vocab.pkl')

    assert ('model_path' in kwargs) ^ ('old_model' in kwargs)

    if 'model_path' in kwargs:
        model = torch.load(kwargs['model_path'])
    else:
        model = torch.load(kwargs['old_model'])

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
    fields = [('sentence', TEXT)]

    dataset = TabularDataset(file_path, format='tsv', skip_header=True, fields=fields)
    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)
    TEXT.vocab = vocab

    device = model.encoder.embedding.weight.device
    batch_size = kwargs.get('batch_size', 64)
    data_iter = Iterator(dataset, batch_size=batch_size, shuffle=False, device=device)

    return sentence_encoding_from_iterator(model, data_iter, **kwargs)