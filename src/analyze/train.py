import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
from src.analyze.iresnet import InvertibleResNet
from src.analyze.dataset import RegressionDataset
from src.analyze.imlp import InvertibleMLP
import os
from copy import deepcopy
from sklearn.linear_model import LinearRegression
import math
from src.analyze.multiple_correlation import multiple_correlation

criterion = nn.MSELoss()

def eval(model, dataloader):
    model.eval()
    total_samples = 0
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            batch_feature, batch_target = data
            batch_feature, batch_target = batch_feature.cuda(), batch_target.cuda()
            batch_prediction = model(batch_feature)
            loss = criterion(batch_prediction, batch_target)
            total_samples += batch_target.size(0)
            total_loss += loss.item() * batch_target.size(0)
    loss = total_loss / total_samples
    return loss


def transform(model, dataloader):
    model.eval()
    feature = []
    transformed_feature = []
    target = []
    with torch.no_grad():
        for data in dataloader:
            batch_feature, batch_target = data
            batch_feature, batch_target = batch_feature.cuda(), batch_target.cuda()
            batch_transformed_feature = model.transform(batch_feature)
            feature.append(batch_feature)
            transformed_feature.append(batch_transformed_feature)
            target.append(batch_target)
    feature = torch.cat(feature, dim=0)
    transformed_feature = torch.cat(transformed_feature, dim=0)
    target = torch.cat(target, dim=0).squeeze(-1)

    return feature, transformed_feature, target


def train():

    os.environ['CUDA_VISIBLE_DEVICES'] = "2"

    dataset = "yelp2"
    base_path = os.path.join("data", dataset)

    batch_size = 100
    lr = 3e-4
    n_blocks = 3

    epoches = 20
    weight_decay = 1e-4
    momentum = 0.9

    train_dataset = RegressionDataset(
        sample_path=os.path.join(base_path, "vanilla_sample_train.tsv"),
        latent_variable_path=os.path.join(base_path, "vanilla_sample_train.npy"),
        targets=["length"]
    )
    dev_dataset = RegressionDataset(
        sample_path=os.path.join(base_path, "vanilla_sample_dev.tsv"),
        latent_variable_path=os.path.join(base_path, "vanilla_sample_dev.npy"),
        targets=["length"]
    )
    test_dataset = RegressionDataset(
        sample_path=os.path.join(base_path, "vanilla_sample_test.tsv"),
        latent_variable_path=os.path.join(base_path, "vanilla_sample_test.npy"),
        targets=["length"]
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    model = InvertibleResNet(
        hidden_size=100,
        target_size=1,
        n_blocks=n_blocks
    )

    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    total_samples = 0
    total_loss = 0

    best_model = None
    min_loss = None

    for epoch in range(epoches):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            batch_feature, batch_target = data
            batch_feature, batch_target = batch_feature.cuda(), batch_target.cuda()

            batch_prediction = model(batch_feature)
            loss = criterion(batch_prediction, batch_target)
            loss.backward()
            optimizer.step()

            total_samples += batch_target.size(0)
            total_loss += loss.item() * batch_target.size(0)

            if i % 100 == 0:
                train_loss = total_loss / total_samples
                total_samples = 0
                total_loss = 0
                dev_loss = eval(model, dev_loader)
                print("[epoch %4d] [step %4d] [train_loss: %.4f] [dev_loss: %.4f]" % (epoch, i, train_loss, dev_loss))
                if min_loss == None or dev_loss < min_loss:
                    min_loss = dev_loss
                    best_model = deepcopy(model)

    test_loss = eval(best_model, test_loader)
    print("test_loss: %.4f" % test_loss)

    path = "iresnet.pkl"
    torch.save(best_model, path)

    feature, transformed_feature, target = transform(best_model, test_loader)
    print(multiple_correlation(feature, target))
    print(multiple_correlation(transformed_feature, target))