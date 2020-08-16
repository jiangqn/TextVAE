import torch
from torch import nn
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import pickle
from src.constants import SOS, EOS, PAD_INDEX

def predict_language_model(model_path, file_path, vocab_path, batch_size=64):

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
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

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX, reduction='none')

    ppl = []
    model.eval()
    with torch.no_grad():

        for batch in test_iter:
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