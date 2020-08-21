import os
import torch
import pickle
from src.utils.sentence_encoding import sentence_encoding_from_tsv
from src.utils.sample_from_encoding import sample_from_encoding
from src.utils.tsv_reader import read_field
from src.utils.encoding_transform import move_encoding

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

    device = model.encoder.embedding.weight.device

    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    with open(principal_directions_save_path, 'rb') as handle:
        principal_directions = pickle.load(handle)

    direction = torch.from_numpy(principal_directions['sentim']).float().to(device)

    with open(projection_statistics_save_path, 'rb') as handle:
        projection_statistics = pickle.load(handle)

    projection_dict = projection_statistics['sentiment']

    current_label = read_field(test_path, 'label')
    target_label = [1 - x for x in current_label]
    target_projection = torch.tensor([projection_dict[x] for x in target_label]).float().to(device)

    original_sentences = read_field(test_path, 'sentence')
    reconstructed_sentences = sample_from_encoding(model, vocab, encoding, config['vae']['batch_size'])

    encoding = move_encoding(encoding, target_projection, direction)

    transferred_sentences = sample_from_encoding(model, vocab, encoding, config['vae']['batch_size'])

    for i, (o, r, t) in enumerate(zip(original_sentences, reconstructed_sentences, transferred_sentences)):
        if i < 50:
            print(i)
            print(o)
            print(r)
            print(t)
            print('')