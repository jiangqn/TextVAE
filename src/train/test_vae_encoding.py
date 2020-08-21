import torch
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import os
import pickle
from src.constants import SOS, EOS, PAD_INDEX

def test_vae_encoding(config: dict) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    save_path = os.path.join(base_path, 'vae.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
    fields = [('sentence', TEXT)]

    test_data = TabularDataset(path=os.path.join(base_path, 'test.tsv'),
                               format='tsv', skip_header=True, fields=fields)
    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)
    TEXT.vocab = vocab

    device = torch.device('cuda:0')
    test_iter = Iterator(test_data, batch_size=config['vae']['batch_size'], shuffle=False, device=device)

    model = torch.load(save_path)