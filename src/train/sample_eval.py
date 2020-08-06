import os
import pickle
import csv
from src.utils import convert_tensor_to_texts
from src.train.predict_language_model import predict_language_model

def sample_eval(model, sample_num=1000, batch_size=64):

    base_path = './data'
    vocab_path = os.path.join(base_path, 'vocab.pkl')
    language_model_path = os.path.join(base_path, 'language_model.pkl')
    sample_save_path = os.path.join(base_path, 'sample_eval.tsv')

    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    batch_sizes = [batch_size] * (sample_num // batch_size) + [sample_num % batch_size]

    sentences = ['sentence']

    for batch_size in batch_sizes:
        output = model.sample(num=batch_size)
        sentences.extend(convert_tensor_to_texts(output, vocab))

    sentences = [[sentence] for sentence in sentences]

    for i in range(1, 6):
        print(sentences[i][0])

    with open(sample_save_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(sentences)

    ppls = predict_language_model(language_model_path, sample_save_path, vocab_path)
    ppl = sum(ppls) / len(ppls)
    os.remove(sample_save_path)
    return ppl