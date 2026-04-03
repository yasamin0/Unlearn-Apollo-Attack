import os
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.svm import SVC

from .attack_framework import Attack_Framework
from models import create_model
import unlearn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class UMIA(Attack_Framework):
    def __init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args):
        super().__init__(target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args)
    def set_include_exclude(self, target_idx):
        pass

    def train_surr(self, surr_idxs, surr_loaders):
        print(">>> U-MIA training")
        X_surr, Y_surr = None, []
        for name, loader in surr_loaders.items():
            for i, (input, label) in enumerate(pbar := tqdm(loader)):
                input, label = input.to(DEVICE), label.to(DEVICE)
                with torch.no_grad():
                    output = self.target_model(input)
                    X_surr = cat(X_surr, output)
                Y_surr.append(int(name == "unlearn"))
        X_surr = X_surr.cpu().numpy()
        Y_surr = np.array(Y_surr)

        self.clf = SVC(C=3, gamma="auto", kernel="rbf", probability=True)
        self.clf.fit(X_surr, Y_surr)

    def update_atk_summary(self, name, target_input, target_label, idx):
        if (not name in self.summary):
            self.summary[name] = dict()

        with torch.no_grad():
            target_output = self.target_model(target_input)

        self.summary[name][idx] = {
            "target_input"  : target_input,
            "target_label"  : target_label,
            "p"             : self.clf.predict_log_proba(target_output.cpu().numpy())
        }
        return None

    def get_ternary_results(self, **kwargs):
        p = {}
        print("Calculating UMIA Ternary Results!")
        
        for name in ["unlearn", "retain", "test"]:
            p[name] = {}
            for i in self.summary[name]:
                p[name][i] = softmax(self.summary[name][i]["p"])

        all_probs = []
        for name in ["unlearn", "retain", "test"]:
            for i in self.summary[name]:
                all_probs.append(p[name][i])
        ths = np.unique(all_probs)
        ternary_points = []
        tpr_results = []  # Store TPRs for each threshold
        accuracy_results = []  # Store overall accuracy for each threshold
        full_classifications = []  # Store full classification details
        
        for th in tqdm(ths):
            classifications = {"unlearn": 0, "retain": 0, "test": 0}
            ground_truth_counts = {"unlearn": 0, "retain": 0, "test": 0}
            correct_classifications = {"unlearn": 0, "retain": 0, "test": 0}
            total_samples = 0
            
            for name in ["unlearn", "retain", "test"]:
                for i in self.summary[name]:
                    ground_truth_counts[name] += 1
                    likelihood_ratio = p[name][i]
                    
                    if likelihood_ratio > th:
                        predicted_class = "unlearn"
                        classifications["unlearn"] += 1
                    else:
                        if name == "test":
                            predicted_class = "test"
                            classifications["test"] += 1
                        else:
                            predicted_class = "retain"
                            classifications["retain"] += 1
                    
                    # Track correct classifications for accuracy
                    if predicted_class == name:
                        correct_classifications[name] += 1
                        
                    total_samples += 1

            if total_samples > 0:
                ternary_point = [
                    classifications["unlearn"] / total_samples,
                    classifications["retain"] / total_samples,
                    classifications["test"] / total_samples
                ]
                ternary_points.append(ternary_point)
                
                # Calculate TPRs for each class
                tpr = {
                    "unlearn": correct_classifications["unlearn"] / ground_truth_counts["unlearn"] if ground_truth_counts["unlearn"] > 0 else 0,
                    "retain": correct_classifications["retain"] / ground_truth_counts["retain"] if ground_truth_counts["retain"] > 0 else 0,
                    "test": correct_classifications["test"] / ground_truth_counts["test"] if ground_truth_counts["test"] > 0 else 0
                }
                tpr_results.append(tpr)
                
                # Calculate overall accuracy
                total_correct = sum(correct_classifications.values())
                overall_accuracy = total_correct / total_samples
                accuracy_results.append(overall_accuracy)
                
                # Store full classification details
                full_classifications.append({
                    'classifications': classifications.copy(),
                    'ground_truth_counts': ground_truth_counts.copy(),
                    'correct_classifications': correct_classifications.copy(),
                    'total_samples': total_samples,
                    'tpr': tpr.copy(),
                    'accuracy': overall_accuracy
                })
        
        results = {
            'ternary_points': np.array(ternary_points),
            'threshold_data': ths,
            'tpr_results': tpr_results,
            'accuracy_results': np.array(accuracy_results),
            'full_classifications': full_classifications
        }
        return results

def cat(A, B) -> torch.Tensor:
    if (A == None):
        return B
    else:
        return torch.cat([A, B], dim=0)

def softmax(output):
    return np.exp(output[0, 1]) / np.sum(np.exp(output))