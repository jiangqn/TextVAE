import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

def correlation(config):

    sample_save_path = input('save path: ')
    # sample_save_path = 'data/sample10000.tsv'
    encoding_save_path = '.'.join(sample_save_path.split('.')[0:-1]) + '.npy'

    df = pd.read_csv(sample_save_path, delimiter='\t')
    sentiment = np.asarray(df['sentiment']).astype(np.float32)[:, np.newaxis]
    ppl = np.asarray(df['ppl']).astype(np.float32)[:, np.newaxis]
    length = np.asarray(df['length']).astype(np.float32)[:, np.newaxis]
    encoding = np.load(encoding_save_path)

    for name, prop in zip(['sentiment', 'ppl', 'length'], [sentiment, ppl, length]):
        cca = CCA(n_components=1)
        u, v = cca.fit_transform(encoding, prop)
        corr = abs(float(np.corrcoef(u[:, 0], v)[0, 1]))
        print('%s correlation: %.4f' % (name, corr))