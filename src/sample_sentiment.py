import os
import torch
import numpy as np
import pickle
import csv
import joblib
from src.constants import SOS, EOS
from src.utils import convert_tensor_to_texts
from src.train.predict_text_cnn import predict_text_cnn

def rejection_sample(num, num_layers, hidden_size, model, label, device):
    encoding = torch.randn(size=(num_layers, num * 5, hidden_size), dtype=torch.float32, device=device)
    transformed_encoding = encoding.transpose(0, 1).reshape(num * 5, -1).cpu().numpy()
    pred_label = model.predict(transformed_encoding)
    index = torch.from_numpy(np.where(pred_label == label)[0]).to(device)
    encoding = encoding.index_select(dim=1, index=index)[:, 0:num, :]
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

def sample_sentiment(config):

    base_path = config['base_path']

    # sample_num = int(input('sample num: '))
    sample_num = 10000
    # sample_save_path = input('save path: ')
    sample_save_path = os.path.join(base_path, 'sample_sentiment10000.tsv')
    # save_encoding = input('save_encoding: ') == 'True'
    save_encoding = True

    assert sample_num % 2 == 0
    sample_positive_num = sample_num // 2
    sample_negative_num = sample_num // 2

    num_layers = config['vae']['num_layers']
    hidden_size = config['vae']['hidden_size']

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    sentiment_lr_path = os.path.join(base_path, 'sentiment_lr.pkl')
    save_path = os.path.join(base_path, 'vae.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')

    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    model = torch.load(save_path)

    sentiment_lr = joblib.load(sentiment_lr_path)

    positive_encoding = rejection_sample(sample_positive_num, num_layers, hidden_size, sentiment_lr, 1, model.encoder.embedding.weight.device)
    negative_encoding = rejection_sample(sample_negative_num, num_layers, hidden_size, sentiment_lr, 0, model.encoder.embedding.weight.device)

    assert positive_encoding.size(1) == sample_positive_num and negative_encoding.size(1) == sample_negative_num

    batch_size = config['vae']['batch_size']

    sentences = ['sentence']

    sentences.extend(sample_from_encoding(model, vocab, positive_encoding, batch_size))
    sentences.extend(sample_from_encoding(model, vocab, negative_encoding, batch_size))

    sentences = [[sentence] for sentence in sentences]
    if save_encoding:
        encoding = torch.cat((positive_encoding, negative_encoding), dim=1)
        encoding = encoding.transpose(0, 1).reshape(sample_num, -1).cpu().numpy()
        encoding_save_path = '.'.join(sample_save_path.split('.')[0:-1]) + '.npy'
        np.save(encoding_save_path, encoding)

    with open(sample_save_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(sentences)

    text_cnn_path = os.path.join(base_path, 'text_cnn.pkl')
    sentiment = predict_text_cnn(text_cnn_path, sample_save_path, vocab_path, batch_size)
    hit = 0
    for i, s in enumerate(sentiment):
        if (i < sample_positive_num and s >= 0.5) or (i >= sample_positive_num and s < 0.5):
            hit += 1
    accuracy = hit / sample_num
    print('sample sentiment accuracy: %.4f' % accuracy)