import os
import torch
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import pickle
from src.constants import SOS, EOS
from src.utils import convert_tensor_to_texts

def sample_from_vae(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    save_path = os.path.join(base_path, 'vae.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')

    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    model = torch.load(save_path)
    output = model.sample(num=100)
    texts = convert_tensor_to_texts(output, vocab)
    for text in texts:
        print(text)