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


def train():

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    dataset = "yelp"
    feature_path = "data/%s/vanilla_sample_100000.npy" % dataset
    target_path = "data/%s/vanilla_sample_100000.tsv" % dataset

    batch_size = 100
    lr = 3e-4
    n_blocks = 1
    epoches = 20
    weight_decay = 1e-3
    momentum = 0.8

    feature = np.load(feature_path)
    target = pd.read_csv(target_path, delimiter="\t")["depth"]

    # lr = LinearRegression()
    # lr.fit(feature[0:80000], target[0:80000])
    # a = lr.predict(feature[90000:100000])
    # n = np.square(a - target[90000: 100000]).mean()
    # print(n)

    target = np.asarray(target)[:, np.newaxis]

    train_dataset = RegressionDataset(feature[0:80000], target[0:80000])
    dev_dataset = RegressionDataset(feature[80000:90000], target[80000:90000])
    test_dataset = RegressionDataset(feature[90000:100000], target[90000:100000])

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

    # old_model = InvertibleResNet(
    #     hidden_size=600,
    #     target_size=1,
    #     n_blocks=n_blocks
    # )

    model = InvertibleMLP(
        hidden_size=600,
        target_size=1
    )

    # feature = torch.from_numpy(feature).double().cuda()
    # target = torch.from_numpy(target).double().cuda()

    model = model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

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

# f = old_model.transform(feature).detach().cpu().numpy()
# t = target[:, 0].cpu().detach().numpy()
# lr = LinearRegression()
# lr.fit(f, t)
# print(epoch, i, math.sqrt(lr.score(f, t)))