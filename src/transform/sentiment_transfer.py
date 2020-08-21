import os
import torch
import pickle
from src.utils.sentence_encoding import sentence_encoding_from_tsv
from src.utils.sample_from_encoding import sample_from_encoding
from src.utils.tsv_reader import read_prop

def sentiment_transfer(config: dict) -> None:

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']

    test_path = os.path.join(base_path, 'dev.tsv')
    vocab_path = os.path.join(base_path, 'vocab.pkl')
    principal_directions_save_path = os.path.join(base_path, 'principal_directions.pkl')
    projection_statistics_save_path = os.path.join(base_path, 'projection_statistics.pkl')
    model_path = os.path.join(base_path, 'vae.pkl')

    model = torch.load(model_path)
    encoding = sentence_encoding_from_tsv(test_path, model=model_path)

    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    with open(principal_directions_save_path, 'rb') as handle:
        principal_directions = pickle.load(handle)

    with open(projection_statistics_save_path, 'rb') as handle:
        projection_statistics = pickle.load(handle)


    transferred_sentences = sample_from_encoding(model, vocab, encoding, config['vae']['batch_size'])
    print(transferred_sentences[0:10])