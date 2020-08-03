import os
import torch
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import pickle
import csv
from src.constants import SOS, EOS
from src.utils import convert_tensor_to_texts

def sample_from_vae(config):

    sample_num = int(input('sample num: '))
    sample_save_path = input('save path: ')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    save_path = os.path.join(base_path, 'vae.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')

    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    model = torch.load(save_path)

    batch_size = config['vae']['batch_size']
    batch_sizes = [batch_size] * (sample_num // batch_size) + [sample_num % batch_size]

    sentences = ['sentence']

    for batch_size in batch_sizes:
        output = model.sample(num=batch_size)
        sentences.extend(convert_tensor_to_texts(output, vocab))

    sentences = [[sentence] for sentence in sentences]

    with open(sample_save_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(sentences)