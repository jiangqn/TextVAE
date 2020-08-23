import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
import os
import pickle
from src.utils.constraint_max_variance_direction_solver import solve

def evaluate_disentanglement(principal_directions):
    keys = list(principal_directions.keys())
    n_keys = len(keys)
    for i, key in enumerate([''] + keys):
        print(key, end='\t\t' if i < n_keys else '\n')
    for key1 in keys:
        print(key1, end='\t\t')
        for j, key2 in enumerate(keys):
            cosine = float(principal_directions[key1].dot(principal_directions[key2]))
            print('%.2f' % cosine, end='\t\t' if j < n_keys - 1 else '\n')

def visualize(config: dict) -> None:

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    vanilla_sample_num = config['vanilla_sample']['sample_num']
    vanilla_sample_save_path = os.path.join(base_path, 'vanilla_sample_%d.tsv' % vanilla_sample_num)
    encoding_save_path = '.'.join(vanilla_sample_save_path.split('.')[0:-1]) + '.npy'
    principal_directions_save_path = os.path.join(base_path, 'principal_directions.pkl')

    encoding = np.load(encoding_save_path)
    num, encoding_size = encoding.shape

    df = pd.read_csv(vanilla_sample_save_path, delimiter='\t')

    encoding_mean = encoding.mean(axis=0)[np.newaxis, :]
    encoding_std = encoding.std(axis=0)[np.newaxis, :]
    encoding = (encoding - encoding_mean) / encoding_std

    prop_names = df.columns.values[2:]

    principal_directions = {}

    for i, prop_name in enumerate(prop_names):

        plt.subplot(2, 2, i + 1)

        prop = np.asarray(df[prop_name]).astype(np.float32)[:, np.newaxis]

        cca = CCA(n_components=1)
        c_encoding, c_prop = cca.fit_transform(encoding, prop)
        v = cca.x_rotations_

        corr = np.corrcoef(c_encoding[:, 0], c_prop)[0, 1]

        A = encoding.transpose().dot(encoding) / num
        u = solve(encoding_size, torch.tensor(A).float(), torch.tensor(v).float()).numpy().astype(np.float64)

        sign = 1 if corr >= 0 else -1
        v = v * sign

        principal_directions[prop_name] = v[:, 0]

        print('%s correlation: %.4f' % (prop_name, sign * corr))
        angle = (np.arccos(np.abs(v[:, 0])) / np.pi * 180).min()
        print('angle: %.4f' % angle)

        x = encoding.dot(v)[:, 0]
        y = encoding.dot(u)[:, 0]

        plt.scatter(x, y, c=prop, s=0.1)
        plt.colorbar()
        plt.title('%s (correlation: %.4f)' % (prop_name, abs(corr)))

    evaluate_disentanglement(principal_directions)

    with open(principal_directions_save_path, 'wb') as handle:
        pickle.dump(principal_directions, handle)

    plt.subplots_adjust(hspace=0.3)
    save_path = os.path.join(base_path, 'visualize.jpg')
    plt.savefig(save_path)