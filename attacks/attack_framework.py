import os
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from collections import OrderedDict
from dataset import PartialDataset

from models import create_model
import unlearn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Attack_Framework():
    def __init__(self, target_model, dataset : PartialDataset, shadow_models : nn.ModuleList, args, idxs: OrderedDict, shadow_col: dict[list], unlearn_args):
        self.target_model = target_model
        self.dataset = dataset
        self.shadow_models = shadow_models
        self.include, self.exclude = [], []
        self.args = args
        self.idxs = idxs
        self.shadow_col = {}
        for i in shadow_col:
            self.shadow_col[i] = set(shadow_col[i])
        self.unlearn_args = unlearn_args

        self.types = [""]
        self.summary = dict()

    def get_unlearned_model(self, i: int):
        unlearned_model = create_model(model_name=self.args.shadow_model, num_classes=self.args.num_classes)
        save_path = os.path.join(
            self.args.shadow_path,
            f"{self.unlearn_args.size_train}",                                      # Target train set
            f"{self.unlearn_args.forget_perc}-{self.unlearn_args.forget_class}",    # Target unlearned set
            f"{self.args.N}",                                                       # Target samples
            f"{self.unlearn_args.unlearn}",                                         # Unlearning method
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        weights_path = os.path.join(save_path, f"{i}.pth.tar")
        cache_path = os.path.join(save_path, f"{i}")

        if os.path.exists(weights_path):
            unlearned_model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
            unlearned_model.to(DEVICE)
            unlearned_model.eval()
        else:
            full = set(self.idxs["unlearn"]).union(set(self.idxs["test"]))
            forget_idx = np.array(list(full.intersection(self.shadow_col[i])), dtype=int)
            retain_idx = np.array(list(set(self.shadow_col[i]).difference(full)), dtype=int)
            # print(">>>", forget_idx[:5], retain_idx[:5])

            forget_set = self.dataset.get_subset(forget_idx)
            retain_set = self.dataset.get_subset(retain_idx)

            forget_loader = DataLoader(forget_set, batch_size=self.unlearn_args.batch_size, shuffle=True, num_workers=4)
            retain_loader = DataLoader(retain_set, batch_size=self.unlearn_args.batch_size, shuffle=True, num_workers=4)

            unlearn_dataloaders = OrderedDict(
                forget_train = forget_loader, retain_train = retain_loader,
                forget_valid = None, retain_valid = None,
            )

            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            ce = nn.CrossEntropyLoss()
            unlearn_method = unlearn.create_unlearn_method(self.unlearn_args.unlearn)(self.shadow_models[i], ce, cache_path, self.unlearn_args)
            unlearn_method.prepare_unlearn(unlearn_dataloaders)
            unlearned_model = unlearn_method.get_unlearned_model()
            torch.save(unlearned_model.state_dict(), weights_path)

        if os.path.exists(cache_path):  # Cleanup
            shutil.rmtree(cache_path)
        return unlearned_model

    def set_include_exclude(self, target_idx):
        include, exclude = [], []
        for i in range(self.args.num_shadow):
            if (target_idx in set(self.shadow_col[i])):
                include.append(i)
            else:
                exclude.append(i)
        if self.args.debug:
            print("target idx:", target_idx, include, exclude)
        self.include, self.exclude = include, exclude

    def update_atk_summary(self, name, target_input, target_label, idx) -> dict:
        return {}
    def get_atk_summary(self):
        summary = self.summary.copy()
        return summary
    def get_roc(self, target_model, **kwargs):
        return

    @staticmethod
    def w(output, label):
        # with torch.no_grad():
        #     w = F.softmax(output, dim=1)[0, label.item()].item()
        # return np.log(w / (1 - w))
        return output[0, label.item()].item()