import os 
import random 
import numpy as np 

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .partial_dataset import PartialDataset

class PartialCIFAR10(PartialDataset):
    def __init__(self, root, img_size=32):
        super().__init__("CIFAR10", root, img_size, "Partial")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_size, antialias=True)])
        self.full_dataset = CIFAR10(root=self.root, train=True,  download=True, transform=transform)
        self.train_len, self.shadow_len, self.valid_len, = 20000, 20000, 10000