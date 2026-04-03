import torch
import torch.nn as nn
from torch.utils.data import Dataset

import copy
import numpy as np

import utils

class PartialDataset:
    def __init__(self, dataset_name, root, img_size, setting="Partial"):
        super().__init__()
        self.dataset_name = dataset_name
        self.root = root
        self.img_size = img_size
        self.train_idx, self.shadow_idx, self.valid_idx = np.array([]), np.array([]), np.array([])
        self.train_len, self.shadow_len, self.valid_len, = 0, 0, 0
        self.full_dataset = Dataset

    def get_subset(self, idx=None) -> Dataset:
        new_dataset = copy.deepcopy(self.full_dataset)
        new_dataset.data = self.full_dataset.data[idx]
        try:
            new_dataset.targets = np.array(self.full_dataset.targets)[idx]
        except:
            new_dataset.labels = np.array(self.full_dataset.labels)[idx]
        return new_dataset

    def set_train_valid_shadow_idx(self, size_train=0, size_shadow=0, num_shadow=0, split="limited", seed=42):
        utils.random_seed(seed)

        N = len(self.full_dataset)
        full_idx = np.arange(N)
        np.random.shuffle(full_idx)

        train_idx  = full_idx[:self.train_len]
        shadow_idx = full_idx[self.train_len:(self.train_len+self.shadow_len)]
        valid_idx  = full_idx[(self.train_len+self.shadow_len):]

        self.train_idx = np.random.choice(train_idx, size=size_train, replace=False)
        self.valid_idx = valid_idx
        self.shadow_col = dict()
        if (split == "limited"):
            for i in range(num_shadow):
                self.shadow_col[i] = np.random.choice(shadow_idx, size=size_shadow, replace=False)
        elif (split == "full"):
            for i in range(num_shadow):
                self.shadow_col[i] = np.random.choice(full_idx, size=size_shadow, replace=False)
        print("train:", len(self.train_idx), self.train_idx[:5])

    def set_unlearn_idx(self, un_perc=None, un_class=None, seed=42):
        utils.random_seed(seed)

        temp_train_idx = self.train_idx
        np.random.shuffle(temp_train_idx)
        if (un_perc != None):
            un_len = int(len(temp_train_idx) * un_perc)
            unlearn_idx, retain_idx = temp_train_idx[:un_len], temp_train_idx[un_len:]

        if (un_class != None):
            try:
                un_mask = np.array(self.full_dataset.targets)[temp_train_idx] == 0
            except:
                un_mask = np.array(self.full_dataset.labels)[temp_train_idx] == 0
            unlearn_idx, retain_idx = temp_train_idx[un_mask], temp_train_idx[np.logical_not(un_mask)]

        print("unlearn:", len(unlearn_idx), unlearn_idx[:5], "retain", len(retain_idx), retain_idx[:5])
        self.unlearn_idx = unlearn_idx
        self.retain_idx = retain_idx