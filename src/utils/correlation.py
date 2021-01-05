import os
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
import pickle
from src.analyze.multiple_correlation import multiple_correlation
import torch

def correlation(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    vanilla_sample_save_path = os.path.join(base_path, 'vanilla_sample_test.tsv')
    encoding_save_path = '.'.join(vanilla_sample_save_path.split('.')[0:-1]) + '.npy'
    principal_directions_save_path = os.path.join(base_path, 'principal_directions.pkl')

    encoding = np.load(encoding_save_path)
    df = pd.read_csv(vanilla_sample_save_path, delimiter='\t')

    encoding_mean = encoding.mean(axis=0)[np.newaxis, :]
    encoding_std = encoding.std(axis=0)[np.newaxis, :]
    encoding = (encoding - encoding_mean) / encoding_std
    encoding = torch.from_numpy(encoding)

    prop_names = ['length', 'depth']

    principal_directions = {}

    for i, prop_name in enumerate(prop_names):

        prop = np.asarray(df[prop_name]).astype(np.float32)[:, np.newaxis]
        prop = torch.from_numpy(prop).squeeze(-1)

        print(prop.size())

        print(multiple_correlation(encoding, prop))

        # cca = CCA(n_components=1)
        # c_encoding, c_prop = cca.fit_transform(encoding, prop)
        # v = cca.x_rotations_
        #
        # corr = np.corrcoef(c_encoding[:, 0], c_prop)[0, 1]
        # sign = 1 if corr >= 0 else -1
        #
        # v = v * sign
        #
        # principal_directions[prop_name] = v[:, 0]
        #
        # print('%s:' % prop_name)
        # print('correlation: %.4f' % (sign * corr))
        # angle = (np.arccos(np.abs(v[:, 0])) / np.pi * 180).min()
        # print('angle between main direction and main dimension: %.4f' % angle)
        # main_dimension = np.argmax(np.abs(v[:, 0]))
        # main_dimension_corr = np.abs(np.corrcoef(encoding[:, main_dimension], c_prop)[0, 1])
        # print('main dimension correlation: %.4f' % main_dimension_corr)
        # print('')

    with open(principal_directions_save_path, 'wb') as handle:
        pickle.dump(principal_directions, handle)