import os
import torch
from torch import nn
from torch import optim
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import numpy as np
import pickle
from src.constants import SOS, EOS, PAD_INDEX

def gradient_encoding(model: nn.Module, sentence: torch.Tensor, **kwargs) -> torch.Tensor:

    lr = kwargs.get('lr', 0.001)
    max_iter = kwargs.get('max_iter', 1000)

    src = sentence[:, 1:]
    trg_input = sentence
    batch_size = sentence.size(0)
    pad = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=sentence.device)
    trg_output = torch.cat((sentence[:, 1:], pad), dim=-1)

    encoding = model.encode(src)
    encoding = nn.Parameter(encoding)

    optimizer = optim.Adam([encoding], lr=lr)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

    model.train()

    min_loss = 1e9
    best_encoding = encoding.data

    for i in range(max_iter):

        optimizer.zero_grad()

        logit = model.decoder(encoding, trg_input)
        trg_output = trg_output.view(-1)
        output_size = logit.size(-1)
        logit = logit.view(-1, output_size)

        loss = criterion(logit, trg_output)
        loss.backward()

        optimizer.step()

        if loss.item() < min_loss:
            min_loss = loss.item()
            best_encoding = encoding.data

    return best_encoding

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
            src = sentence[:, 1:]

            if probabilistic_encoding:
                batch_encoding, _, _ = model.probabilistic_encode(src)
            else:
                batch_encoding, _ = model.encode(src)

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

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
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