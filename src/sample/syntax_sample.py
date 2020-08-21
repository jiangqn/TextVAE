import os
import torch
import numpy as np
import pickle
import csv
from src.utils.tsv_reader import read_prop
from src.utils.convert_tensor_to_texts import convert_tensor_to_texts
from src.utils.encoding_transform import interpolate
from src.utils.sample_from_encoding import sample_from_encoding

def syntax_sample(config: dict, prop_name: str) -> None:

    scope = config['%s_sample' % prop_name]['scope']
    intervals = config['%s_sample' % prop_name]['intervals']

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
    direction = torch.from_numpy(principal_directions[prop_name]).float().to(device)
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