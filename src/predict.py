import pandas as pd
import os
from src.train.predict_text_cnn import predict_text_cnn
from src.train.predict_language_model import predict_language_model

def predict(config):

    base_path = config['base_path']

    # sample_save_path = input('save path: ')
    sample_save_path = os.path.join(base_path, 'sample10000.tsv')
    # encoding_save_path = '.'.join(sample_save_path.split('.')[0:-1]) + '.npy'

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    vocab_path = os.path.join(base_path, 'vocab.pkl')
    text_cnn_path = os.path.join(base_path, 'text_cnn.pkl')
    language_model_path = os.path.join(base_path, 'language_model.pkl')

    sentiment = predict_text_cnn(text_cnn_path, sample_save_path, vocab_path, batch_size=config['text_cnn']['batch_size'])
    ppl = predict_language_model(language_model_path, sample_save_path, vocab_path, batch_size=config['language_model']['batch_size'])

    df = pd.read_csv(sample_save_path, delimiter='\t')
    length = [len(sentence.split()) for sentence in list(df.loc[:, 'sentence'])]
    from src.sentence_depth import sentence_depth
    depth = [sentence_depth(sentence) for sentence in list(df.loc[:, 'sentence'])]
    df['sentiment'] = sentiment
    df['ppl'] = ppl
    df['length'] = length
    df['depth'] = depth
    df.to_csv(sample_save_path, sep='\t')
