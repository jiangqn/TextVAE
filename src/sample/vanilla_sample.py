import os
import torch
import numpy as np
import pickle
import csv
from src.utils.convert_tensor_to_texts import convert_tensor_to_texts
from src.get_features.get_ppl import get_ppl_from_tsv
from src.utils import metric
from src.train.eval_reverse_ppl import eval_reverse_ppl

def vanilla_sample(config: dict) -> None:

    base_path = config['base_path']

    vanilla_sample_num = config['vanilla_sample']['sample_num']
    vanilla_sample_save_path = os.path.join(base_path, 'vanilla_sample_%d.tsv' % vanilla_sample_num)
    save_encoding = config['vanilla_sample']['save_encoding']

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    save_path = os.path.join(base_path, 'vae.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')
    language_model_path = os.path.join(base_path, 'language_model.pkl')

    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    model = torch.load(save_path)

    batch_size = config['vae']['batch_size']
    batch_sizes = [batch_size] * (vanilla_sample_num // batch_size) + [vanilla_sample_num % batch_size]

    sentences = ['sentence']

    if save_encoding:
        encoding = []

    for batch_size in batch_sizes:
        if save_encoding:
            output, output_encoding = model.sample(num=batch_size, output_encoding=True)
            encoding.append(output_encoding)
        else:
            output = model.sample(num=batch_size)
        sentences.extend(convert_tensor_to_texts(output, vocab))

    sentences = [[sentence] for sentence in sentences]
    if save_encoding:
        encoding = np.concatenate(encoding, axis=0)
        encoding_save_path = '.'.join(vanilla_sample_save_path.split('.')[0:-1]) + '.npy'
        np.save(encoding_save_path, encoding)

    with open(vanilla_sample_save_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(sentences)

    ppl = get_ppl_from_tsv(vanilla_sample_save_path, config['language_model']['batch_size'], model_path=language_model_path, vocab_path=vocab_path)

    print('vanilla sample ppl: %.4f' % metric.mean(ppl))