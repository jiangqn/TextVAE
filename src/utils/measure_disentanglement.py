import os
import numpy as np
import pandas as pd

def measure_disentanglement(config: dict) -> None:

    base_path = config['base_path']
    vanilla_sample_num = config['vanilla_sample']['sample_num']
    vanilla_sample_save_path = os.path.join(base_path, 'vanilla_sample_%d.tsv' % vanilla_sample_num)
    df = pd.read_csv(vanilla_sample_save_path, delimiter='\t')
    label = np.asarray(df['label'])
    length = np.asarray(df['length'])
    depth = np.asarray(df['depth'])

    index = np.where(label <= 1)[0]
    label = label[index]
    length = length[index]
    depth = depth[index]

    named_features = [('label', label), ('length', length), ('depth', depth)]
    features = [named_feature[1] for named_feature in named_features]

    print('corr\tlabel\tlength\tdepth')

    for (name1, feature1) in named_features:
        print(name1, end='')
        for feature2 in features:
            corr = np.corrcoef(feature1, feature2)[0, 1]
            print('\t%.2f' % corr, end='')
        print('')