import os
import torch
import pickle
from src.utils.tsv_reader import read_field
from src.utils.multinomial_distribution import get_multinomial_distribution, sample_from_multinomial_distribution
from src.utils.encoding_transform import move_encoding
from src.utils.sample_from_encoding import sample_from_encoding
from src.get_features.get_length import get_length
from src.utils import metric

def length_sample(config: dict) -> None:

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']

    sample_num = 10000

    sample_save_path = os.path.join(base_path, 'length_sample%d.tsv' % sample_num)
    vanilla_sample_save_path = os.path.join(base_path, 'sample10000.tsv')

    vocab_path = os.path.join(base_path, 'vocab.pkl')
    principal_directions_save_path = os.path.join(base_path, 'principal_directions.pkl')
    projection_statistics_save_path = os.path.join(base_path, 'projection_statistics.pkl')
    model_path = os.path.join(base_path, 'vae.pkl')

    vanilla_sample_length = read_field(vanilla_sample_save_path, 'length')
    length_distribution = get_multinomial_distribution(vanilla_sample_length)
    target_length = sample_from_multinomial_distribution(length_distribution, sample_num)

    model = torch.load(model_path)
    device = model.encoder.embedding.weight.device
    num_layers = model.num_layers
    hidden_size = model.hidden_size

    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    with open(principal_directions_save_path, 'rb') as handle:
        principal_directions = pickle.load(handle)

    direction = torch.from_numpy(principal_directions['length']).float().to(device)

    with open(projection_statistics_save_path, 'rb') as handle:
        projection_statistics = pickle.load(handle)

    projection_dict = projection_statistics['length']
    target_projection = torch.tensor([projection_dict[x] for x in target_length]).float().to(device)

    encoding = torch.randn(size=(num_layers, sample_num, hidden_size), device=device)
    encoding = move_encoding(encoding, target_projection, direction)

    sentences = sample_from_encoding(model, vocab, encoding, config['vae']['batch_size'])
    length = get_length(sentences)

    print('accuracy: %.4f' % metric.accuracy(length, target_length))
    print('rmse: %.4f' % metric.rmse(length, target_length))
    print('diff: %.4f' % metric.diff(length, target_length))