import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np

class ClassificationDataset(Dataset):

    def __init__(self, feature: np.ndarray, label: np.ndarray) -> None:
        self.feature = torch.from_numpy(feature).float()
        self.label = torch.from_numpy(label).long()
        self.num = self.label.size(0)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.feature[item], self.label[item]

def get_classification_dataloader(feature: np.ndarray, label: np.ndarray, batch_size: int = 64, shuffle: bool = False, split: bool = False):

    assert feature.shape[0] == label.shape[0]

    if split:
        num = feature.shape[0]
        train_num = int(0.8 * num)

        train_feature, train_label = feature[0: train_num], label[0: train_num]
        dev_feature, dev_label = feature[train_num: num], label[train_num: num]

        train_loader = get_classification_dataloader(train_feature, train_label, batch_size=batch_size, shuffle=shuffle)
        dev_loader = get_classification_dataloader(dev_feature, dev_label, batch_size=batch_size)

        return train_loader, dev_loader
    else:
        dataset = ClassificationDataset(feature, label)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True
        )

        return dataloader