import os
import torch
import numpy as np
import pickle
import joblib
import csv
from src.utils.multinomial_distribution import get_multinomial_distribution_from_tsv
from src.utils.rejection_sample import multinomial_rejection_sample
from src.utils.sample_from_encoding import sample_from_encoding
from src.get_features.get_category import get_categorical_features_from_tsv
from src.get_features.get_ppl import get_ppl_from_tsv
from src.utils import metric

def categorical_sample(config: dict) -> None:

    base_path = config['base_path']
    vanilla_sample_num = config['vanilla_sample']['sample_num']
    vanilla_sample_save_path = os.path.join(base_path, 'vanilla_sample_%d.tsv' % vanilla_sample_num)
    categorical_sample_num = config['categorical_sample']['sample_num']
    categorical_sample_save_path = os.path.join(base_path, 'categorical_sample_%d.tsv' % categorical_sample_num)
    save_encoding = config['categorical_sample']['save_encoding']

    num_layers = config['vae']['num_layers']
    hidden_size = config['vae']['hidden_size']

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    classifier_path = os.path.join(base_path, 'classifier.pkl')
    model_path = os.path.join(base_path, 'vae.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')
    text_cnn_path = os.path.join(base_path, 'text_cnn.pkl')
    language_model_path = os.path.join(base_path, 'language_model.pkl')

    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    model = torch.load(model_path)

    classifier = joblib.load(classifier_path)

    label_distribution = get_multinomial_distribution_from_tsv(vanilla_sample_save_path, 'label')
    target_label, encoding = multinomial_rejection_sample(num_layers, categorical_sample_num, hidden_size, classifier, label_distribution)

    device = model.encoder.embedding.weight.device
    encoding = encoding.to(device)

    sentences = ['sentence'] + sample_from_encoding(model, vocab, encoding, config['max_len'], config['vae']['batch_size'])

    sentences = [[sentence] for sentence in sentences]
    if save_encoding:
        encoding = encoding.transpose(0, 1).reshape(categorical_sample_num, -1).cpu().numpy()
        encoding_save_path = '.'.join(categorical_sample_save_path.split('.')[0:-1]) + '.npy'
        np.save(encoding_save_path, encoding)

    with open(categorical_sample_save_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(sentences)

    label = get_categorical_features_from_tsv(file_path=categorical_sample_save_path, batch_size=config['text_cnn']['batch_size'],
                                              model_path=text_cnn_path, vocab_path=vocab_path, output_category=True)

    print('category sample accuracy: %.4f' % metric.accuracy(label, target_label.tolist()))

    ppl = get_ppl_from_tsv(file_path=categorical_sample_save_path, batch_size=config['language_model']['batch_size'],
                           model_path=language_model_path, vocab_path=vocab_path)

    print('category sample ppl: %.4f' % metric.mean(ppl))