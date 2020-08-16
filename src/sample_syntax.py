import os
import torch
import numpy as np
import pickle
import csv
import joblib
from src.constants import SOS, EOS
from src.utils import convert_tensor_to_texts

def interpolate(encoding, direction, interval):
    pass

def sample_from_encoding(model, vocab, encoding, batch_size):
    sample_num = encoding.size(0)
    sentences = []
    start = 0
    while start < sample_num:
        end = min(sample_num, start + batch_size)
        output = model.sample(encoding=encoding[:, start:end, :])
        sentences.extend(convert_tensor_to_texts(output, vocab))
        start = end
    return sentences

def sample_syntax(config):

    prop = 'length'

    base_path = config['base_path']

    # sample_num = int(input('sample num: '))
    sample_num = 10000
    # sample_save_path = input('save path: ')
    sample_save_path = os.path.join(base_path, 'sample_length100.tsv')
    # save_encoding = input('save_encoding: ') == 'True'
    save_encoding = True

    num_layers = config['vae']['num_layers']
    hidden_size = config['vae']['hidden_size']

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    save_path = os.path.join(base_path, 'vae.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')
    principal_directions_save_path = os.path.join(base_path, 'principal_directions.pkl')

    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    with open(principal_directions_save_path, 'rb') as handle:
        principal_directions = pickle.load(handle)

    model = torch.load(save_path)

    length_direction = torch.from_numpy(principal_directions['length'])

    batch_size = config['vae']['batch_size']

    sentences = ['sentence']

    # sentences.extend(sample_from_encoding(model, vocab, positive_encoding, batch_size))
    # sentences.extend(sample_from_encoding(model, vocab, negative_encoding, batch_size))
    #
    # sentences = [[sentence] for sentence in sentences]
    # if save_encoding:
    #     encoding = np.concatenate((positive_encoding, negative_encoding), axis=0)
    #     encoding_save_path = '.'.join(sample_save_path.split('.')[0:-1]) + '.npy'
    #     np.save(encoding_save_path, encoding)
    #
    # with open(sample_save_path, 'w') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerows(sentences)