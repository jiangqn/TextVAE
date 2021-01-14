import os
import torch
import pickle
import csv
from src.utils.tsv_reader import read_field
from src.utils.multinomial_distribution import get_multinomial_distribution, sample_from_multinomial_distribution
from src.utils.encoding_transform import move_encoding
from src.utils.sample_from_encoding import sample_sentences_from_latent_variable
from src.get_features.get_depth import get_depth
from src.utils import metric
from src.get_features.get_ppl import get_ppl_from_tsv
from src.train.eval_reverse_ppl import eval_reverse_ppl

def linear_depth_sample(config: dict) -> None:

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    vanilla_sample_num = config['vanilla_sample']['sample_num']
    vanilla_sample_save_path = os.path.join(base_path, 'vanilla_sample_%d.tsv' % vanilla_sample_num)

    depth_sample_num = config['depth_sample']['sample_num']
    depth_sample_save_path = os.path.join(base_path, 'depth_sample_%d.tsv' % depth_sample_num)

    vocab_path = os.path.join(base_path, 'vocab.pkl')
    principal_directions_save_path = os.path.join(base_path, 'principal_directions.pkl')
    projection_statistics_save_path = os.path.join(base_path, 'projection_statistics.pkl')
    model_path = os.path.join(base_path, 'vae.pkl')
    language_model_path = os.path.join(base_path, 'language_model.pkl')

    vanilla_sample_depth = read_field(vanilla_sample_save_path, 'depth')
    depth_distribution = get_multinomial_distribution(vanilla_sample_depth)
    target_depth = sample_from_multinomial_distribution(depth_distribution, depth_sample_num)

    model = torch.load(model_path)
    device = model.encoder.embedding.weight.device
    num_layers = model.num_layers
    hidden_size = model.hidden_size

    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    with open(principal_directions_save_path, 'rb') as handle:
        principal_directions = pickle.load(handle)

    direction = torch.from_numpy(principal_directions['depth']).float().to(device)

    with open(projection_statistics_save_path, 'rb') as handle:
        projection_statistics = pickle.load(handle)

    projection_dict = projection_statistics['depth']
    target_projection = torch.tensor([projection_dict[x] for x in target_depth]).float().to(device)

    encoding = torch.randn(size=(num_layers, depth_sample_num, hidden_size), device=device)
    encoding = move_encoding(encoding, target_projection, direction)

    sentences = sample_sentences_from_latent_variable(model, vocab, encoding, config['max_len'], config['vae']['batch_size'])
    depth = get_depth(sentences, processes=20)

    sentences = ['sentence'] + sentences
    sentences = [[sentence] for sentence in sentences]

    with open(depth_sample_save_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(sentences)

    ppl = get_ppl_from_tsv(depth_sample_save_path, config['language_model']['batch_size'], model_path=language_model_path, vocab_path=vocab_path)

    print('depth sample')
    print('accuracy: %.4f' % metric.accuracy(depth, target_depth))
    print('diff: %.4f' % metric.diff(depth, target_depth))
    print('correlation: %.4f' % metric.correlation(depth, target_depth))
    print('ppl: %.4f' % metric.mean(ppl))

    reverse_ppl = eval_reverse_ppl(config, depth_sample_save_path)

    print('depth sample reverse ppl: %.4f' % reverse_ppl)