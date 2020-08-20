import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def vanilla_visualize(config: dict, method: str) -> None:

    assert method in ['pca', 'tsne']

    base_path = config['base_path']
    sample_save_path = os.path.join(base_path, 'sample10000.tsv')
    encoding_save_path = '.'.join(sample_save_path.split('.')[0:-1]) + '.npy'

    encoding = np.load(encoding_save_path)

    df = pd.read_csv(sample_save_path, delimiter='\t')

    if method == 'pca':
        model = PCA(n_components=2)
    else: # tsne
        model = TSNE(n_components=2)

    encoding = model.fit_transform(encoding)

    prop_names = df.columns.values[2:]

    for i, prop_name in enumerate(prop_names):
        plt.subplot(2, 2, i + 1)

        prop = np.asarray(df[prop_name]).astype(np.float32)[:, np.newaxis]

        x = encoding[:, 0]
        y = encoding[:, 1]

        plt.scatter(x, y, c=prop, s=0.1)
        plt.colorbar()
        plt.title('%s' % prop_name)

    plt.subplots_adjust(hspace=0.3)
    save_path = os.path.join(base_path, '%s_visualize.jpg' % method)
    plt.savefig(save_path)