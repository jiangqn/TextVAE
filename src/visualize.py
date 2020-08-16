import torch
from torch import nn, optim
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
import os
import pickle

class Solver(nn.Module):

    def __init__(self, m, lambd=-1, mu=-1):
        super(Solver, self).__init__()
        self.u = nn.Parameter(torch.rand(m, 1))
        self.lambd = lambd
        self.mu = mu
        self.solution = None

    def forward(self, A, v):
        return - (self.u.t().matmul(A).matmul(self.u)[0, 0] + self.lambd * torch.abs(self.u.t().matmul(self.u) - 1) + self.mu * torch.abs(self.u.t().matmul(v)))

def solve(m, A, v):

    solver = Solver(m, -10, -10)

    optimizer = optim.SGD(solver.parameters(), lr=0.0003, momentum=0.9)

    n_step = 10000

    min_loss = 1e9
    for i in range(n_step):
        optimizer.zero_grad()
        loss = solver(A, v)
        loss.backward()
        if i % 10 == 0:
            # print('%d\t%.4f' % (i, loss.item()))
            if loss.item() < min_loss:
                min_loss = loss.item()
                solver.solution = deepcopy(solver.u.data)
        optimizer.step()
    # print(min_loss)
    # print(solver.solution.t().matmul(solver.solution).item())
    # print(solver.solution.t().matmul(v).item())
    return solver.solution

def evaluate_disentanglement(principal_directions):
    keys = list(principal_directions.keys())
    n_keys = len(keys)
    for i, key in enumerate([''] + keys):
        print(key, end='\t' if i < n_keys else '\n')
    for key1 in keys:
        print(key1, end='\t')
        for j, key2 in enumerate(keys):
            cosine = float(principal_directions[key1].dot(principal_directions[key2]))
            print('%.2f' % cosine, end='\t' if j < n_keys - 1 else '\n')

def visualize(config):

    base_path = config['base_path']

    sample_save_path = os.path.join(base_path, 'sample10000.tsv')
    encoding_save_path = '.'.join(sample_save_path.split('.')[0:-1]) + '.npy'
    principal_directions_save_path = os.path.join(base_path, 'principal_directions.pkl')

    encoding = np.load(encoding_save_path)
    n, m = encoding.shape

    df = pd.read_csv(sample_save_path, delimiter='\t')

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

        principal_directions[prop_name[0:6]] = v[:, 0]

        corr = np.corrcoef(c_encoding[:, 0], c_prop)[0, 1]

        A = encoding.transpose().dot(encoding) / n
        u = solve(m, torch.tensor(A).float(), torch.tensor(v).float()).numpy().astype(np.float64)

        sign = 1 if corr >= 0 else -1

        print('%s correlation: %.4f' % (prop_name, sign * corr))

        x = encoding.dot(v)[:, 0] * sign
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