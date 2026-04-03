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

from .attack_framework import Attack_Framework
from models import create_model
import unlearn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ULiRA(Attack_Framework):
    def __init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args):
        super().__init__(target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args)
        self.unlearned_shadow_models = nn.ModuleList()
        for i in range(self.args.num_shadow):
            self.unlearned_shadow_models.append( self.get_unlearned_model(i) )
        # exit()

    def update_atk_summary(self, name, target_input, target_label, idx):
        if (not name in self.summary):
            self.summary[name] = dict()
        w_in, w_ex = [], []
        for i in self.include:
            model = self.unlearned_shadow_models[i]
            with torch.no_grad():
                target_output = model(target_input)
            w_in.append(self.w(target_output, target_label))
        for i in self.exclude:
            model = self.shadow_models[i]
            with torch.no_grad():
                target_output = model(target_input)
            w_ex.append(self.w(target_output, target_label))
        self.summary[name][idx] = {
            "target_input"      : target_input,
            "target_label"      : target_label,
            "w_in"      : w_in,
            "w_ex"      : w_ex,
        }
        return None
    
    def get_ternary_results(self, **kwargs):
        p = {}
        print("Calculating ULiRA Ternary Results!")
        
        for name in ["unlearn", "retain", "test"]:
            p[name] = {}
            for i in self.summary[name]:
                with torch.no_grad():
                    target_output = self.target_model(self.summary[name][i]["target_input"])
                target_w = self.w(target_output, self.summary[name][i]["target_label"])
                if (len(self.summary[name][i]["w_in"]) == 0) or (len(self.summary[name][i]["w_ex"]) == 0):
                    p[name][i] = np.log(1)  # Neutral likelihood ratio
                else:
                    p[name][i] = np.log(pr(target_w, self.summary[name][i]["w_in"]) / (pr(target_w, self.summary[name][i]["w_ex"]) + 1e-9))

        all_ratios = []
        for name in ["unlearn", "retain", "test"]:
            for i in self.summary[name]:
                all_ratios.append(p[name][i])
        ths = np.unique(all_ratios)
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

def pr(x, obs):
    mean, std = norm.fit(obs)
    return norm.pdf(x, mean, std)