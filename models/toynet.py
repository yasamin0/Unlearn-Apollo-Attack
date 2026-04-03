import torch
import torch.nn as nn
import torch.nn.functional as F

class toynet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(2, 256)
        hidden_layers = []
        for _ in range(10):
            hidden_layers.append(nn.Linear(256, 256))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.BatchNorm1d(256))
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.hidden(out)
        out = self.output(out)
        return out

def ToyNet(num_classes):
    return toynet(num_classes)