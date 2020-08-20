import torch
from torch import nn
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import pickle
from typing import List
from src.constants import SOS, EOS, PAD_INDEX

def get_ppl(model: nn.Module, data_iter: Iterator) -> List[float]:

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX, reduction='none')

    ppl = []

    model.eval()
    with torch.no_grad():

        for batch in data_iter:

            sentence = batch.sentence
            input_sentence = sentence
            batch_size = sentence.size(0)
            pad = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=sentence.device)
            output_sentence = torch.cat((sentence[:, 1:], pad), dim=-1)

            logit = model(input_sentence)
            output_sentence = output_sentence.view(-1)
            output_size = logit.size(-1)
            logit = logit.view(-1, output_size)
            loss = criterion(logit, output_sentence)
            loss = loss.reshape(batch_size, -1)
            output_sentence = output_sentence.reshape(batch_size, -1)
            mask = (output_sentence != PAD_INDEX).float()
            output_sentence_lens = mask.sum(dim=1, keepdim=False)
            batch_ppl = torch.pow(2, (loss * mask).sum(dim=1, keepdim=False) / output_sentence_lens)
            ppl.extend(batch_ppl.tolist())

    return ppl

def get_ppl_from_tsv(file_path: str, batch_size: int = 64, **kwargs) -> List[float]:

    assert ('model_path' in kwargs) ^ ('model' in kwargs)
    assert ('vocab_path' in kwargs) ^ ('vocab' in kwargs)

    if 'model_path' in kwargs:
        model = torch.load(kwargs['model_path'])
    else:
        model = kwargs['model']

    if 'vocab_path' in kwargs:
        with open(kwargs['vocab_path'], 'rb') as handle:
            vocab = pickle.load(handle)
    else:
        vocab = kwargs['vocab']

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
    fields = [
        ('sentence', TEXT)
    ]

    test_data = TabularDataset(path=file_path, format='tsv', skip_header=True, fields=fields)
    TEXT.vocab = vocab

    device = torch.device('cuda:0')
    test_iter = Iterator(test_data, batch_size=batch_size, shuffle=False, device=device)

    return get_ppl(model, test_iter)