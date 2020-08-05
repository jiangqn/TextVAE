import torch
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import pickle

def predict_text_cnn(model_path, file_path, vocab_path, batch_size=64):

    TEXT = data.Field(sequential=True, lower=True, batch_first=True)
    fields = [
        ('sentence', TEXT)
    ]

    test_data = TabularDataset(path=file_path, format='tsv', skip_header=True, fields=fields)
    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)
    TEXT.vocab = vocab

    device = torch.device('cuda:0')
    test_iter = Iterator(test_data, batch_size=batch_size, shuffle=False, device=device)
    model = torch.load(model_path)

    sentiments = []
    model.eval()
    with torch.no_grad():

        for batch in test_iter:
            sentence = batch.sentence
            logit = model(sentence)
            prob = torch.softmax(logit, dim=-1)[:, 1].tolist()
            sentiments.extend(prob)

    return sentiments