import pandas as pd
import os
import math
from src.get_features.get_category import get_categorical_features_from_tsv
from src.get_features.get_ppl import get_ppl_from_tsv
from src.get_features.get_length import get_length
from src.get_features.get_depth import get_depth

def get_features(config: dict) -> None:

    base_path = config['base_path']

    # sample_save_path = input('save path: ')
    sample_save_path = os.path.join(base_path, 'sample10000.tsv')
    # encoding_save_path = '.'.join(sample_save_path.split('.')[0:-1]) + '.npy'

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    vocab_path = os.path.join(base_path, 'vocab.pkl')
    text_cnn_path = os.path.join(base_path, 'text_cnn.pkl')
    language_model_path = os.path.join(base_path, 'language_model.pkl')

    print('get categorical features')
    sentiment = get_categorical_features_from_tsv(file_path=sample_save_path, batch_size=config['text_cnn']['batch_size'],
        model_path=text_cnn_path, vocab_path=vocab_path, output_score=True)

    print('get ppl')
    ppl = get_ppl_from_tsv(file_path=sample_save_path, batch_size=config['language_model']['batch_size'],
        model_path=language_model_path, vocab_path=vocab_path)

    df = pd.read_csv(sample_save_path, delimiter='\t')
    sentences = list(df['sentence'])

    print('get length')
    length = get_length(sentences)

    print('get depth')
    depth = get_depth(sentences, processes=20)

    df['sentiment'] = sentiment
    df['logppl'] = [math.log2(x) for x in ppl]
    df['length'] = length
    df['depth'] = depth

    df.to_csv(sample_save_path, sep='\t')
