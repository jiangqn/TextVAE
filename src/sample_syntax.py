import os
import torch
import numpy as np
import pickle
import csv
import joblib
from src.tsv_reader import read_prop
from src.utils import convert_tensor_to_texts

def interpolate(encoding, direction, scope, intervals):
    sample_num = encoding.size(1)
    hidden_size = encoding.size(2)
    encoding = encoding.transpose(0, 1).reshape(sample_num, -1)
    projection = encoding.matmul(direction)
    start_encoding = encoding + (-scope - projection) * direction.transpose(0, 1)
    end_encoding = encoding + (scope - projection) * direction.transpose(0, 1)
    weights = torch.arange(0, 1, 1 / intervals, device=encoding.device)
    start_encoding = start_encoding.unsqueeze(1).repeat(1, intervals, 1)
    end_encoding = end_encoding.unsqueeze(1).repeat(1, intervals, 1)
    weights = weights.unsqueeze(0).unsqueeze(-1)
    encoding = start_encoding * (1 - weights) + end_encoding * weights
    encoding = encoding.reshape(sample_num * intervals, -1)
    encoding = encoding.reshape(sample_num * intervals, 1, hidden_size).transpose(0, 1)
    return encoding

def sample_from_encoding(model, vocab, encoding, batch_size):
    sample_num = encoding.size(1)
    sentences = []
    start = 0
    while start < sample_num:
        end = min(sample_num, start + batch_size)
        output = model.sample(encoding=encoding[:, start:end, :])
        sentences.extend(convert_tensor_to_texts(output, vocab))
        start = end
    return sentences

def sample_syntax(config):

    prop_name = 'depth'
    scope = 2
    intervals = 10

    base_path = config['base_path']

    # sample_num = int(input('sample num: '))
    sample_num = 1000
    # sample_save_path = input('save path: ')
    sample_save_path = os.path.join(base_path, 'sample_%s%d.tsv' % (prop_name, sample_num))
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
    device = model.encoder.embedding.weight.device
    batch_size = config['vae']['batch_size']

    encoding = torch.randn(size=(num_layers, sample_num, hidden_size), device=device)
    direction = torch.from_numpy(principal_directions[prop_name]).unsqueeze(-1).float().to(device)
    encoding = interpolate(encoding, direction, scope, intervals)

    sentences = ['sentence']

    sentences.extend(sample_from_encoding(model, vocab, encoding, batch_size))

    sentences = [[sentence] for sentence in sentences]
    if save_encoding:
        encoding = encoding.transpose(0, 1).reshape(sample_num * intervals, -1).cpu().numpy()
        encoding_save_path = '.'.join(sample_save_path.split('.')[0:-1]) + '.npy'
        np.save(encoding_save_path, encoding)

    with open(sample_save_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(sentences)

    prop = read_prop(sample_save_path, prop_name)
    projection = np.asarray(torch.arange(-scope, scope, 2 * scope / intervals).tolist() * sample_num)

    corr = float(np.corrcoef(projection, prop)[0, 1])
    print('%s correlation: %.4f' % (prop_name, corr))