import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset

from .partial_dataset import PartialDataset

import pickle as pkl
import utils

def generate_dataset(N, num_classes=4):
    utils.random_seed()
    dataset, labels = [], []

    for i in range(N):
        for _ in range(125):
            a = np.random.uniform( (2 * np.pi) / N * i, (2 * np.pi) / N * (i + 1) )
            r = np.random.uniform(0.1, 1)
            dataset.append([r * np.cos(a), r * np.sin(a)])
            labels.append((i % num_classes))
    return np.array(dataset), np.array(labels)

class ToyDataset(Dataset):
    """Dataset class for ImageNet"""
    def __init__(self, dataset, labels, train=False):
        super(ToyDataset, self).__init__()
        assert(len(dataset) == len(labels))
        self.data = dataset
        self.targets = labels

        self.transform = None
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.data)[idx], torch.Tensor(self.targets)[idx]

class PartialToyData(PartialDataset):
    def __init__(self, root, img_size=0):
        super().__init__("ToyData", root, img_size, "Partial")
        dataset, labels = generate_dataset(N=4, num_classes=4)
        self.full_dataset = ToyDataset(dataset, labels, train=True)
        self.train_len, self.shadow_len, self.valid_len, = 200, 200, 100