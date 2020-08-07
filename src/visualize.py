import torch
from torch import nn, optim
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
import os

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

def visualize(config):

    base_path = config['base_path']

    sample_save_path = 'data/sample10000.tsv'
    encoding_save_path = '.'.join(sample_save_path.split('.')[0:-1]) + '.npy'

    encoding = np.load(encoding_save_path)
    n, m = encoding.shape

    df = pd.read_csv(sample_save_path, delimiter='\t')

    encoding_mean = encoding.mean(axis=0)[np.newaxis, :]
    encoding_std = encoding.std(axis=0)[np.newaxis, :]
    encoding = (encoding - encoding_mean) / encoding_std

    prop_names = df.columns.values[2:]

    for i, prop_name in enumerate(prop_names):

        plt.subplot(2, 2, i + 1)

        prop = np.asarray(df[prop_name]).astype(np.float32)[:, np.newaxis]

        cca = CCA(n_components=1)
        c_encoding, c_prop = cca.fit_transform(encoding, prop)
        v = cca.x_rotations_

        corr = np.corrcoef(c_encoding[:, 0], c_prop)[0, 1]

        A = encoding.transpose().dot(encoding) / n
        u = solve(m, torch.tensor(A).float(), torch.tensor(v).float()).numpy().astype(np.float64)

        sign = 1 if corr >= 0 else -1

        x = encoding.dot(v)[:, 0] * sign
        y = encoding.dot(u)[:, 0]

        plt.scatter(x, y, c=prop)
        plt.colorbar()
        plt.title('%s (correlation: %.4f)' % (prop_name, abs(corr)))

    plt.subplots_adjust(hspace=0.3)
    save_path = os.path.join(base_path, 'visualize.jpg')
    plt.savefig(save_path)