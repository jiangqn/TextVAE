import torch
from torch import nn
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import pickle

def get_categorical_features(model: nn.Module, data_iter: Iterator, **kwargs) -> list:

    output_category = kwargs.get('output_category', False)
    output_score = kwargs.get('output_score', False)
    assert output_category ^ output_score

    categories = []
    scores = []

    model.eval()
    with torch.no_grad():

        for batch in data_iter:

            sentence = batch.sentence
            logit = model(sentence)

            batch_categories = logit.argmax(dim=-1)
            categories.extend(batch_categories.tolist())

            num_categories = logit.size(1)
            base = torch.arange(num_categories, dtype=torch.float, device=logit.device).unsqueeze(-1)
            batch_prob = torch.softmax(logit, dim=-1)
            batch_scores = batch_prob.matmul(base)[:, 0]
            scores.extend(batch_scores.tolist())

    if output_category:
        return categories
    else:
        return scores


def get_categorical_features_from_tsv(file_path, batch_size, **kwargs) -> list:

    assert ('model_path' in kwargs) ^ ('old_model' in kwargs)
    assert ('vocab_path' in kwargs) ^ ('vocab' in kwargs)

    if 'model_path' in kwargs:
        model = torch.load(kwargs['model_path'])
    else:
        model = kwargs['old_model']

    if 'vocab_path' in kwargs:
        with open(kwargs['vocab_path'], 'rb') as handle:
            vocab = pickle.load(handle)
    else:
        vocab = kwargs['vocab']

    TEXT = data.Field(sequential=True, lower=True, batch_first=True)
    fields = [
        ('sentence', TEXT)
    ]

    test_data = TabularDataset(path=file_path, format='tsv', skip_header=True, fields=fields)
    TEXT.vocab = vocab

    device = torch.device('cuda:0')
    test_iter = Iterator(test_data, batch_size=batch_size, shuffle=False, device=device)

    return get_categorical_features(model, test_iter, **kwargs)