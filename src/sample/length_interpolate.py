import os
import torch
import pickle
import csv
from src.utils.tsv_reader import read_field
from src.utils.multinomial_distribution import get_multinomial_distribution, sample_from_multinomial_distribution
from src.utils.encoding_transform import move_encoding
from src.utils.sample_from_encoding import sample_sentences_from_latent_variable
from src.utils.encoding_transform import interpolate

def length_interpolate(config: dict) -> None:

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']

    length_interpolate_sample_num = 1000
    intervals = 5

    length_interpolate_sample_save_path = os.path.join(base_path, 'length_interpolate.tsv')

    vocab_path = os.path.join(base_path, 'vocab.pkl')
    principal_directions_save_path = os.path.join(base_path, 'principal_directions.pkl')
    projection_statistics_save_path = os.path.join(base_path, 'projection_statistics.pkl')
    model_path = os.path.join(base_path, 'vae.pkl')

    model = torch.load(model_path)
    device = model.encoder.embedding.weight.device
    num_layers = model.num_layers
    hidden_size = model.hidden_size

    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    with open(principal_directions_save_path, 'rb') as handle:
        principal_directions = pickle.load(handle)

    direction = torch.from_numpy(principal_directions['length']).float().to(device)

    encoding = torch.randn(size=(num_layers, length_interpolate_sample_num, hidden_size), device=device)
    encoding = interpolate(encoding, direction, scope=2, intervals=intervals)

    sentences = sample_sentences_from_latent_variable(model, vocab, encoding, config['max_len'], config['vae']['batch_size'])

    sentences = ['sentence'] + sentences
    sentences = [[sentence] for sentence in sentences]

    with open(length_interpolate_sample_save_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(sentences)