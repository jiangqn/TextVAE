import os
from src.analyze.multiple_correlation import multiple_correlation
import numpy as np
import pandas as pd
import torch

dataset = "ptb"
feature_name = "depth"
division = "train"

latent_variable_path = os.path.join("data", dataset, "vanilla_sample_%s.npy" % division)
feature_path = os.path.join("data", dataset, "vanilla_sample_%s.tsv" % division)

latent_variable = np.load(latent_variable_path)
feature = np.asarray(pd.read_csv(feature_path, delimiter="\t")[feature_name])
target = np.asarray(pd.read_csv(feature_path, delimiter="\t")["length"])

print(np.corrcoef(feature, target))

latent_variable = torch.from_numpy(latent_variable).float()
feature = torch.from_numpy(feature).float()

print(multiple_correlation(latent_variable, feature))