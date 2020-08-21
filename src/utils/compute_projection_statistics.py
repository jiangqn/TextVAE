import os
import numpy as np
import pandas as pd
import pickle

def compute_projection_statistics(config: dict) -> None:

    base_path = config['base_path']

    sample_save_path = os.path.join(base_path, 'sample10000.tsv')
    encoding_save_path = '.'.join(sample_save_path.split('.')[0:-1]) + '.npy'
    principal_directions_save_path = os.path.join(base_path, 'principal_directions.pkl')
    projection_statistics_save_path = os.path.join(base_path, 'projection_statistics.pkl')

    df = pd.read_csv(sample_save_path, delimiter='\t')
    sentiment = np.asarray(df['sentiment'])
    length = np.asarray(df['length'])
    depth = np.asarray(df['depth'])

    encoding = np.load(encoding_save_path)
    with open(principal_directions_save_path, 'rb') as handle:
        principal_directions = pickle.load(handle)

    sentiment_projection = encoding.dot(principal_directions['sentim'])
    length_projection = encoding.dot(principal_directions['length'])
    depth_projection = encoding.dot(principal_directions['depth'])

    projection_statistics = {
        'sentiment': {},
        'length': {},
        'depth': {}
    }

    projection_statistics['sentiment'] = {
        0: float(sentiment_projection[np.where(sentiment < 0.1)[0]].mean()),
        1: float(sentiment_projection[np.where(sentiment > 0.9)[0]].mean())
    }

    projection_statistics['length'] = {
        i: float(length_projection[np.where(length == i)[0]].mean()) for i in set(list(length))
    }

    projection_statistics['depth'] = {
        i: float(depth_projection[np.where(depth == i)[0]].mean()) for i in set(list(depth))
    }

    with open(projection_statistics_save_path, 'wb') as handle:
        pickle.dump(projection_statistics, handle)