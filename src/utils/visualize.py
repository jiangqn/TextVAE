import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import os
import pickle
import joblib
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

def numerical_visualize(base_path: str, encoding: np.ndarray, df: pd.DataFrame) -> None:

    principal_directions_save_path = os.path.join(base_path, 'principal_directions.pkl')

    with open(principal_directions_save_path, 'rb') as handle:
        principal_directions = pickle.load(handle)

    num, dim = encoding.shape
    covariance = encoding.transpose().dot(encoding) / num

    cmap = {
        'length': 'viridis',
        'depth': 'cividis'
    }

    # pdf = PdfPages(os.path.join(base_path, 'numerical_visualize.pdf'))
    plt.figure()

    for i, prop in enumerate(['length', 'depth']):

        v = principal_directions[prop]
        u = solve(torch.tensor(covariance).float(), torch.tensor(v).float()).cpu().numpy().astype(np.float64)[:, 0]

        x = encoding.dot(v)
        y = encoding.dot(u)

        feature = np.asarray(df[prop])

        plt.subplot(2, 2, i + 1)
        plt.scatter(x, y, c=feature, s=0.1, cmap=cmap[prop])

        corr = np.corrcoef(x, feature)[0, 1]
        plt.title('%s (correlation: %.4f)' % (prop, corr))
        plt.colorbar()

        pca = PCA(n_components=2)
        pca_encoding = pca.fit_transform(encoding)

        plt.subplot(2, 2, i + 3)
        plt.scatter(pca_encoding[:, 0], pca_encoding[:, 1], c=feature, s=0.1, cmap=cmap[prop])
        plt.title('%s (PCA)' % prop)
        plt.colorbar()

    plt.subplots_adjust(hspace=0.3)
    # pdf.savefig()
    plt.savefig(os.path.join(base_path, 'numerical_visualize.png'))
    plt.close()
    # pdf.close()


def categorical_visualize(base_path: str, encoding: np.ndarray, df: pd.DataFrame) -> None:

    prop = 'sentiment' if 'yelp' in base_path else 'topic'

    cmap = {
        'sentiment': 'summer',
        'topic': 'winter'
    }

    classifier_path = os.path.join(base_path, 'classifier.pkl')

    model = joblib.load(classifier_path).model

    label = np.asarray(df['label'])

    index = np.where(label <= 1)[0]
    encoding = encoding[index]
    label = label[index]

    num, dim = encoding.shape
    covariance = encoding.transpose().dot(encoding) / num

    cca = CCA(n_components=1)
    c_encoding, c_prop = cca.fit_transform(encoding, label)
    v = cca.x_rotations_

    corr = np.corrcoef(c_encoding[:, 0], c_prop)[0, 1]

    sign = 1 if corr >= 0 else -1
    v = v[:, 0] * sign
    u = solve(torch.tensor(covariance).float(), torch.tensor(v).float()).cpu().numpy().astype(np.float64)[:, 0]

    x = encoding.dot(v)
    y = encoding.dot(u)

    # pdf = PdfPages(os.path.join(base_path, 'categorical_visualize.pdf'))
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)

    plt.scatter(x, y, c=label, s=0.1, cmap=cmap[prop])

    w = (model.linear.weight[0, :] - model.linear.weight[1, :]).cpu().detach().numpy().astype(np.float64)
    wx = w.dot(v)
    wy = w.dot(u)
    rg = lambda a, b, n: [a + (b - a) / n * i for i in range(n)]
    intervals = 1000
    plt.scatter(rg(wy / wx * 4, -wy / wx * 4, intervals), rg(4, -4, intervals), c='black', s=0.1)

    plt.title('%s' % prop)

    pca = PCA(n_components=2)
    pca_encoding = pca.fit_transform(encoding)

    plt.subplot(1, 2, 2)
    plt.scatter(pca_encoding[:, 0], pca_encoding[:, 1], c=label, s=0.1, cmap=cmap[prop])
    plt.title('%s (PCA)' % prop)

    # pdf.savefig()
    plt.savefig(os.path.join(base_path, 'categorical_visualize.png'))
    plt.close()
    # pdf.close()

def visualize(config: dict) -> None:

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    vanilla_sample_num = config['vanilla_sample']['sample_num']
    vanilla_sample_save_path = os.path.join(base_path, 'vanilla_sample_%d.tsv' % vanilla_sample_num)
    encoding_save_path = '.'.join(vanilla_sample_save_path.split('.')[0:-1]) + '.npy'
    classifier_path = os.path.join(base_path, 'classifier.pkl')


    encoding = np.load(encoding_save_path)
    num, encoding_size = encoding.shape

    df = pd.read_csv(vanilla_sample_save_path, delimiter='\t')

    encoding_mean = encoding.mean(axis=0)[np.newaxis, :]
    encoding_std = encoding.std(axis=0)[np.newaxis, :]
    encoding = (encoding - encoding_mean) / encoding_std

    categorical_visualize(base_path, encoding, df)
    numerical_visualize(base_path, encoding, df)

    # classifier = joblib.load(classifier_path)
    # print(classifier.linear.weight.shape)
    #
    # prop_names = df.columns.values[2:]
    #
    # principal_directions = {}
    #
    # pdf = PdfPages(os.path.join(base_path, 'visualize.pdf'))
    #
    # plt.figure()

    # for i, prop_name in enumerate(prop_names):
    #
    #     plt.subplot(2, 2, i + 1)
    #
    #     prop = np.asarray(df[prop_name]).astype(np.float32)[:, np.newaxis]
    #
    #     cca = CCA(n_components=1)
    #     c_encoding, c_prop = cca.fit_transform(encoding, prop)
    #     v = cca.x_rotations_
    #
    #     corr = np.corrcoef(c_encoding[:, 0], c_prop)[0, 1]
    #
    #     A = encoding.transpose().dot(encoding) / num
    #     u = solve(torch.tensor(A).float(), torch.tensor(v).float()).cpu().numpy().astype(np.float64)
    #
    #     sign = 1 if corr >= 0 else -1
    #     v = v * sign
    #
    #     principal_directions[prop_name] = v[:, 0]
    #
    #     print('%s correlation: %.4f' % (prop_name, sign * corr))
    #     angle = (np.arccos(np.abs(v[:, 0])) / np.pi * 180).min()
    #     print('angle: %.4f' % angle)
    #
    #     x = encoding.dot(v)[:, 0]
    #     y = encoding.dot(u)[:, 0]
    #
    #     plt.scatter(x, y, c=prop, s=0.1)
    #     plt.colorbar()
    #     plt.title('%s (correlation: %.4f)' % (prop_name, abs(corr)))
    #
    # evaluate_disentanglement(principal_directions)

    # with open(principal_directions_save_path, 'wb') as handle:
    #     pickle.dump(principal_directions, handle)

    # plt.subplots_adjust(hspace=0.3)
    # # save_path = os.path.join(base_path, 'visualize.jpg')
    # # plt.savefig(save_path)
    # pdf.savefig()
    # plt.close()
    # pdf.close()