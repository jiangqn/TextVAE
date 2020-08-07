import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

def correlation(config):

    # sample_save_path = input('save path: ')
    sample_save_path = 'data/sample10000.tsv'
    encoding_save_path = '.'.join(sample_save_path.split('.')[0:-1]) + '.npy'

    encoding = np.load(encoding_save_path)
    df = pd.read_csv(sample_save_path, delimiter='\t')

    prop_names = df.columns.values[2:]

    for prop_name in prop_names:
        cca = CCA(n_components=1)
        prop = np.asarray(df[prop_name]).astype(np.float32)[:, np.newaxis]
        u, v = cca.fit_transform(encoding, prop)
        corr = abs(float(np.corrcoef(u[:, 0], v)[0, 1]))
        print('%s correlation: %.4f' % (prop_name, corr))