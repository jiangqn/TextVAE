import torch
import os
from sklearn.linear_model import LinearRegression
import math
import numpy as np
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

dataset = "yelp"
feature_path = "../data/%s/vanilla_sample_100000.npy" % dataset
target_path = "../data/%s/vanilla_sample_100000.tsv" % dataset

feature = np.load(feature_path)
target = pd.read_csv(target_path, delimiter="\t")["depth"]
target = np.asarray(target)[:, np.newaxis]

feature = torch.from_numpy(feature).double()
target = torch.from_numpy(target).double()

path = "iresnet.pkl"
model = torch.load(path, map_location=torch.device('cpu'))
model.check()

latent = model.transform(feature)
backfeature = model.inverse_transform(latent)

norm = torch.norm(feature - backfeature, dim=1).mean().item()
print(norm)