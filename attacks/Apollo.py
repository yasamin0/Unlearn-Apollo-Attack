import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import stack_module_state, functional_call

from .attack_framework import Attack_Framework

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Apollo(Attack_Framework):
    def __init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args):
        super().__init__(target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args)
        self.types = ["Unified"]  # Single unified attack type
        self.unlearned_shadow_models = nn.ModuleList()
        for i in range(self.args.num_shadow):
            self.unlearned_shadow_models.append(self.get_unlearned_model(i))

    def set_include_exclude(self, target_idx):
        super().set_include_exclude(target_idx)
        if (len(self.include)):
            self.temp_un, self.params_un, self.buffers_un = batched_models_(
                [self.unlearned_shadow_models[i] for i in self.include]
            )
        if (len(self.exclude)):
            self.temp_rt, self.params_rt, self.buffers_rt = batched_models_(
                [self.shadow_models[i] for i in self.exclude]
            )


    def batched_loss_Under(self, input, label):
        loss_un, loss_rt = 0., 0.
        if (len(self.include)):
            loss_un = batched_loss_(input, label, self.temp_un, self.params_un, self.buffers_un)
        if (len(self.exclude)):            
            loss_rt = batched_loss_(input, label, self.temp_rt, self.params_rt, self.buffers_rt)
        return self.args.w[0] * loss_un - self.args.w[1] * loss_rt
    
    def batched_loss_Over(self, input, label):
        loss_un, loss_rt = 0., 0.
        if (len(self.include)):
            loss_un = batched_loss_(input, label, self.temp_un, self.params_un, self.buffers_un)
        if (len(self.exclude)):            
            loss_rt = batched_loss_(input, label, self.temp_rt, self.params_rt, self.buffers_rt)
        return -(self.args.w[0] * loss_un + self.args.w[1] * loss_rt)

    def Un_Adv(self, target_input: torch.Tensor, target_label: torch.Tensor, loss_func):
        """
        Adversarial perturbation generation following Apollo theoretical framework:
        - Under-unlearning: Creates inputs that expose incomplete forgetting
        - Over-unlearning: Creates inputs that expose excessive forgetting
        """
        adv_input = target_input.detach().clone().to(DEVICE)
        adv_input.requires_grad = True
        optimizer = torch.optim.Adam([adv_input], lr=self.args.atk_lr)  # Adam for better convergence
        
        conf, pred, target_conf = [], [], []
        
        for epoch in range(self.args.atk_epochs):
            optimizer.zero_grad()
            loss = loss_func(adv_input, target_label)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta = adv_input.data - target_input
                delta = torch.clamp(delta, -self.args.eps, self.args.eps)
                adv_input.data = target_input + delta
                adv_input.data.clamp_(0.0, 1.0)

            with torch.no_grad():
                target_output = self.target_model(adv_input)
                target_pred = target_output.max(1)[1].item()
                pred.append(target_pred)
                # target_logit = target_output[0, target_label.item()].item()
                # target_conf.append(target_logit)
                shadow_conf = 0.
                if len(self.exclude) > 0:
                    for i in self.exclude:
                        shadow_output = self.shadow_models[i](adv_input)
                        shadow_logit = shadow_output[0, target_label.item()].item()
                        shadow_conf += shadow_logit
                    shadow_conf /= len(self.exclude)
                conf.append(-shadow_conf if len(self.exclude) > 0 else 0)
                
        return conf, pred

    def update_atk_summary(self, name, target_input, target_label, idx):
        if (not name in self.summary):
            self.summary[name] = dict()
        un_conf, un_pred = self.Un_Adv(target_input, target_label, self.batched_loss_Under)
        ov_conf, ov_pred = self.Un_Adv(target_input, target_label, self.batched_loss_Over)
        self.summary[name][idx] = {
            "target_input"  : target_input,
            "target_label"  : target_label,
            "un_conf"       : un_conf,
            "un_pred"       : un_pred,
            "ov_conf"       : ov_conf,
            "ov_pred"       : ov_pred,
        }
        return None

    def get_ternary_results(self, **kwargs):
        print("Calculating Apollo Ternary Results!")
        
        gt, under_conf, over_conf, under_pred, over_pred = {}, {}, {}, {}, {}
        
        for name in ["unlearn", "retain", "test"]:
            gt[name] = torch.cat([self.summary[name][i]["target_label"] for i in self.summary[name]], dim=0).cpu().numpy()
            under_conf[name] = np.array([self.summary[name][i]["un_conf"] for i in self.summary[name]])
            over_conf[name] = np.array([self.summary[name][i]["ov_conf"] for i in self.summary[name]])
            under_pred[name] = np.array([self.summary[name][i]["un_pred"] for i in self.summary[name]])
            over_pred[name] = np.array([self.summary[name][i]["ov_pred"] for i in self.summary[name]])

        # Use all confidence values across all epochs for comprehensive threshold space
        all_under_vals = []
        all_over_vals = []
        for name in ["unlearn", "retain", "test"]:
            for i in range(len(under_conf[name])):
                all_under_vals.extend(under_conf[name][i])
                all_over_vals.extend(over_conf[name][i])
        
        under_ths = np.percentile(all_under_vals, np.linspace(0, 100, 21))  # Use percentiles for better coverage
        over_ths = np.percentile(all_over_vals, np.linspace(0, 100, 21))
        
        ternary_points = []
        threshold_pairs = []
        tpr_results = []  # Store TPRs for each threshold pair
        accuracy_results = []  # Store overall accuracy for each threshold pair
        full_classifications = []  # Store full classification details
        
        for under_th in tqdm(under_ths):
            for over_th in over_ths:
                # Classification counts and ground truth tracking
                classifications = {"unlearn": 0, "retain": 0, "test": 0}
                ground_truth_counts = {"unlearn": 0, "retain": 0, "test": 0}
                correct_classifications = {"unlearn": 0, "retain": 0, "test": 0}
                total_samples = 0
                
                for name in ["unlearn", "retain", "test"]:
                    for i in range(len(under_conf[name])):
                        true_label = gt[name][i]
                        ground_truth_counts[name] += 1
                        
                        # Apollo methodology: Find the epoch with maximum attack effectiveness
                        under_scores = []
                        over_scores = []
                        
                        for epoch in range(self.args.atk_epochs):
                            # Under-unlearning score: higher confidence indicates under-unlearning
                            under_score = under_conf[name][i][epoch] - under_th
                            under_correct = (under_pred[name][i][epoch] == true_label)
                            # Penalize incorrect predictions in under-unlearning (should maintain correctness)
                            if not under_correct:
                                under_score -= 1.0
                            under_scores.append(under_score)
                            
                            # Over-unlearning score: lower confidence indicates over-unlearning vulnerability
                            over_score = over_th - over_conf[name][i][epoch] 
                            over_correct = (over_pred[name][i][epoch] == true_label)
                            # Reward incorrect predictions in over-unlearning detection
                            if not over_correct:
                                over_score += 1.0
                            over_scores.append(over_score)
                        
                        # Find best epochs for each attack type
                        best_under_epoch = np.argmax(under_scores)
                        best_over_epoch = np.argmax(over_scores)
                        
                        max_under_score = under_scores[best_under_epoch]
                        max_over_score = over_scores[best_over_epoch]
                        
                        # Determine attack success based on theoretical framework
                        under_success = max_under_score > 0
                        over_success = max_over_score > 0
                        
                        if under_success and over_success:
                            # Both attacks successful - choose stronger signal
                            primary_under = max_under_score > max_over_score
                        elif under_success:
                            primary_under = True
                        elif over_success:
                            primary_under = False
                        else:
                            # Neither attack successful - apply theoretical defaults
                            if name == "retain":
                                predicted_class = "retain"
                                classifications["retain"] += 1  # Default: under-unlearned
                            elif name == "test":
                                predicted_class = "test"
                                classifications["test"] += 1    # Default: over-unlearned baseline
                            else:  # unlearn
                                predicted_class = "unlearn"
                                classifications["unlearn"] += 1  # Default: properly unlearned
                            
                            # Track correct classifications for accuracy
                            if predicted_class == name:
                                correct_classifications[name] += 1
                            
                            total_samples += 1
                            continue
                        
                        # Classification based on Apollo's theoretical framework
                        if primary_under:
                            # Under-unlearning detection succeeded
                            epoch = best_under_epoch
                            is_correct = (under_pred[name][i][epoch] == true_label)
                            
                            # High confidence + correct prediction = retained (under-unlearned)
                            # High confidence + wrong prediction = over-unlearned side effect  
                            if is_correct and under_conf[name][i][epoch] > under_th:
                                predicted_class = "retain"
                                classifications["retain"] += 1
                            else:
                                predicted_class = "unlearn"
                                classifications["unlearn"] += 1
                                
                        else:
                            # Over-unlearning detection succeeded
                            epoch = best_over_epoch
                            is_correct = (over_pred[name][i][epoch] == true_label)
                            
                            # Low confidence or wrong prediction = test baseline (over-unlearned)
                            # High confidence + correct = properly learned
                            if not is_correct or over_conf[name][i][epoch] < over_th:
                                predicted_class = "test"
                                classifications["test"] += 1
                            else:
                                predicted_class = "unlearn"
                                classifications["unlearn"] += 1
                        
                        # Track correct classifications for accuracy
                        if predicted_class == name:
                            correct_classifications[name] += 1
                            
                        total_samples += 1
                
                # Convert to proportions for ternary plot
                if total_samples > 0:
                    ternary_point = [
                        classifications["unlearn"] / total_samples,
                        classifications["retain"] / total_samples, 
                        classifications["test"] / total_samples
                    ]
                    ternary_points.append(ternary_point)
                    threshold_pairs.append((under_th, over_th))
                    
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
            'threshold_data': np.array(threshold_pairs),
            'tpr_results': tpr_results,
            'accuracy_results': np.array(accuracy_results),
            'full_classifications': full_classifications
        }
        return results

class Apollo_Offline(Apollo):
    def __init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args):
        Attack_Framework.__init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args)
        self.types = ["Unified"]  # Single unified attack type

    def set_include_exclude(self, target_idx):
        Attack_Framework.set_include_exclude(self, target_idx)
        self.temp, self.params, self.buffers = batched_models_(self.shadow_models)

    def batched_loss_Under(self, input, label):
        outputs = torch.vmap(
            lambda ps, bs, xx: functional_call(self.temp, ps, (xx,), {}, tie_weights=True, strict=False),
            in_dims=(0, 0, None)
        )(self.params, self.buffers, input)

        flat = outputs.reshape(-1, outputs.size(-1))
        label_rep = label.repeat(outputs.size(0))

        loss_rt = F.cross_entropy(flat, label_rep)
        top2_vals, _ = outputs.topk(2, dim=-1)
        loss_db = top2_vals[0, :, 0] - top2_vals[0, :, 1]
        return self.args.w[0] * loss_db - self.args.w[1] * loss_rt
    
    def batched_loss_Over(self, input, label):
        outputs = torch.vmap(
            lambda ps, bs, xx: functional_call(self.temp, ps, (xx,), {}, tie_weights=True, strict=False),
            in_dims=(0, 0, None)
        )(self.params, self.buffers, input)

        flat = outputs.reshape(-1, outputs.size(-1))
        label_rep = label.repeat(outputs.size(0))

        loss_rt = F.cross_entropy(flat, label_rep)
        top2_vals, _ = outputs.topk(2, dim=-1)
        loss_db = top2_vals[0, :, 0] - top2_vals[0, :, 1]
        return self.args.w[0] * loss_db + self.args.w[1] * loss_rt


def batched_models_(models_list):
    temp = models_list[0]
    params, buffers = stack_module_state(models_list)
    return temp, params, buffers

def batched_loss_(input, label, temp, params, buffers):
    # outputs: (N, batch, classes)
    outputs = torch.vmap(
        lambda ps, bs, xx: functional_call(temp, ps, (xx,), {}, tie_weights=True, strict=False),
        in_dims=(0, 0, None)
    )(params, buffers, input)

    flat = outputs.reshape(-1, outputs.size(-1))
    label_rep = label.repeat(outputs.size(0))
    return F.cross_entropy(flat, label_rep)

def proj(A: torch.Tensor, B: torch.Tensor, r: float, type):
    with torch.no_grad():
        d = (B - A).view(-1).norm(p=2).item()
        if (type == "in"):
            scale = min(1.0, r / (d + 1e-9))
        else:
            scale = max(1.0, r / (d + 1e-9))
        return A + (B - A) * scale