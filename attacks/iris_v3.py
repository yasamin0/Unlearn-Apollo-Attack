import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call

from .Apollo import Apollo_Offline

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IRIS_V3(Apollo_Offline):
    """
    IRIS_v3:
    - Apollo backbone
    - radius-guided initialization
    - relative scoring
    - early-trajectory features

    This class keeps Apollo's dual-channel structure and ternary evaluation flow,
    but modifies the adversarial search and the stored signal construction.
    """

    def __init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args):
        super().__init__(target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args)
        self.types = ["Unified"]

        # Radius-guided search args
        self.iris_probe_radius = float(getattr(args, "iris_probe_radius", 0.10))
        self.iris_probe_steps = int(getattr(args, "iris_probe_steps", 8))
        self.iris_probe_samples = int(getattr(args, "iris_probe_samples", 12))

        # Relative scoring / early-trajectory args
        self.iris_use_relative_score = bool(getattr(args, "iris_use_relative_score", True))
        self.iris_use_early_features = bool(getattr(args, "iris_use_early_features", True))
        self.iris_early_k = int(getattr(args, "iris_early_k", 5))
        self.iris_min_epochs = int(getattr(args, "iris_min_epochs", 10))
        self.iris_patience = int(getattr(args, "iris_patience", 4))
    
    def _shadow_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batched shadow forward pass.
        Output shape: [num_shadow, batch, num_classes]
        """
        outputs = torch.vmap(
            lambda ps, bs, xx: functional_call(
                self.temp, ps, (xx,), {}, tie_weights=True, strict=False
            ),
            in_dims=(0, 0, None)
        )(self.params, self.buffers, x)
        return outputs

    def _compute_mode_loss_and_shadow_logit(self, x: torch.Tensor, target_label: torch.Tensor, mode: str):
        """
        Single batched shadow pass that gives:
        - Apollo-style loss for optimization
        - mean shadow logit on the true class for relative scoring
        """
        outputs = self._shadow_outputs(x)  # [num_shadow, batch, num_classes]

        flat = outputs.reshape(-1, outputs.size(-1))
        label_rep = target_label.repeat(outputs.size(0))

        loss_rt = F.cross_entropy(flat, label_rep)
        top2_vals, _ = outputs.topk(2, dim=-1)
        loss_db = top2_vals[0, :, 0] - top2_vals[0, :, 1]

        if mode == "under":
            loss = self.args.w[0] * loss_db - self.args.w[1] * loss_rt
        elif mode == "over":
            loss = self.args.w[0] * loss_db + self.args.w[1] * loss_rt
        else:
            raise ValueError(f"Unknown mode: {mode}")

        cls = target_label.item()
        mean_shadow_logit = outputs[:, 0, cls].mean()

        return loss, mean_shadow_logit

    def _cheap_probe_score(self, x: torch.Tensor, target_label: torch.Tensor) -> float:
        """
        Cheap shared probe score:
        target true-class logit minus mean shadow true-class logit.
        Uses one target forward + one batched shadow forward.
        """
        with torch.no_grad():
            self._qa_target(1)
            t_out = self.target_model(x)

            shadow_count = len(self.shadow_models)
            self._qa_shadow(shadow_count)
            s_outputs = self._shadow_outputs(x)

            cls = target_label.item()
            t_logit = t_out[0, cls].item()
            s_logit = s_outputs[:, 0, cls].mean().item()

        return float(t_logit - s_logit)  

    def _radius_probe(self, target_input: torch.Tensor, target_label: torch.Tensor):
        """
        Shared cheap probe:
        sample a few random local directions and choose the best start point
        using a cheap relative target-vs-shadow score.
        """
        best_x = target_input.detach().clone()
        best_score = None
        base_score = None

        with torch.no_grad():
            base_score = self._cheap_probe_score(best_x, target_label)
            best_score = base_score

            for _ in range(self.iris_probe_samples):
                noise = torch.randn_like(target_input)
                noise_norm = noise.view(-1).norm(p=2).item() + 1e-12
                direction = noise / noise_norm

                candidate = target_input + self.iris_probe_radius * direction
                candidate = torch.clamp(candidate, 0.0, 1.0)

                cand_score = self._cheap_probe_score(candidate, target_label)

                if cand_score > best_score:
                    best_score = cand_score
                    best_x = candidate.detach().clone()

        return best_x, float(best_score - base_score)

    def _early_features(self, traj):
        """
        Compute early-trajectory summary stats.
        """
        if len(traj) == 0:
            return {
                "early_mean": 0.0,
                "early_max": 0.0,
                "early_gain": 0.0,
                "full_max": 0.0,
            }

        k = min(self.iris_early_k, len(traj))
        early = traj[:k]

        return {
            "early_mean": float(np.mean(early)),
            "early_max": float(np.max(early)),
            "early_gain": float(early[-1] - early[0]) if len(early) >= 2 else 0.0,
            "full_max": float(np.max(traj)),
        }
    def _epoch_monitor_score(self, conf_value: float, pred_value: int, true_label: int, mode: str) -> float:
        """
        Score used only for early stopping.
        Higher is better.
        """
        score = float(conf_value)

        if mode == "under":
            # under branch prefers correct prediction with strong confidence
            if pred_value == true_label:
                score += 1.0
        elif mode == "over":
            # over branch prefers label instability / misclassification signal
            if pred_value != true_label:
                score += 1.0

        return float(score)

    def IRIS_Adv(self, target_input: torch.Tensor, target_label: torch.Tensor, mode: str, init_x: torch.Tensor = None, probe_gain: float = 0.0):
        """
        Modified Apollo search with:
        1) shared radius-guided initialization
        2) single batched shadow forward per epoch
        3) no duplicate target-shadow gap queries
        4) adaptive early stopping
        """
        if init_x is None:
            init_x, probe_gain = self._radius_probe(target_input, target_label)

        adv_input = init_x.detach().clone().to(DEVICE)
        adv_input.requires_grad = True
        optimizer = optim.Adam([adv_input], lr=self.args.atk_lr)

        conf = []
        pred = []
        rel_score = []

        best_monitor = -float("inf")
        epochs_without_improve = 0

        for epoch in range(self.args.atk_epochs):
            optimizer.zero_grad()

            self._qa_steps(1)

            shadow_count = len(self.shadow_models)
            self._qa_shadow(shadow_count)
            loss, mean_shadow_logit = self._compute_mode_loss_and_shadow_logit(
                adv_input, target_label, mode=mode
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta = adv_input.data - target_input
                delta = torch.clamp(delta, -self.args.eps, self.args.eps)
                adv_input.data = target_input + delta
                adv_input.data.clamp_(0.0, 1.0)

            with torch.no_grad():
                self._qa_target(1)
                target_output = self.target_model(adv_input)
                target_pred = target_output.max(1)[1].item()
                pred.append(target_pred)

                target_logit = target_output[0, target_label.item()].item()
                current_rel_gap = float(target_logit - mean_shadow_logit.item())

                if self.iris_use_relative_score:
                    current_conf = current_rel_gap
                else:
                    current_conf = float(target_logit)

                conf.append(current_conf)
                rel_score.append(current_rel_gap)

                # -------- early stopping monitor --------
                monitor = self._epoch_monitor_score(
                    conf_value=current_conf,
                    pred_value=target_pred,
                    true_label=target_label.item(),
                    mode=mode,
                )

                if monitor > best_monitor + 1e-12:
                    best_monitor = monitor
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1

                if (epoch + 1) >= self.iris_min_epochs and epochs_without_improve >= self.iris_patience:
                    break

        early = self._early_features(conf if self.iris_use_relative_score else rel_score)

        return {
            "conf": conf,
            "pred": pred,
            "probe_gain": float(probe_gain),
            "early_mean": float(early["early_mean"]),
            "early_max": float(early["early_max"]),
            "early_gain": float(early["early_gain"]),
            "full_max": float(early["full_max"]),
            "mode": mode,
            "epochs_used": int(len(conf)),
        }

    def update_atk_summary(self, name, target_input, target_label, idx):
        if name not in self.summary:
            self.summary[name] = dict()

        self._qa_start(name, idx)

        shared_init_x, shared_probe_gain = self._radius_probe(target_input, target_label)

        under = self.IRIS_Adv(
            target_input=target_input,
            target_label=target_label,
            mode="under",
            init_x=shared_init_x,
            probe_gain=shared_probe_gain,
        )

        over = self.IRIS_Adv(
            target_input=target_input,
            target_label=target_label,
            mode="over",
            init_x=shared_init_x,
            probe_gain=shared_probe_gain,
        )

        self.summary[name][idx] = {
            "target_input": target_input,
            "target_label": target_label,

            "un_conf": under["conf"],
            "un_pred": under["pred"],
            "ov_conf": over["conf"],
            "ov_pred": over["pred"],

            "un_probe_gain": under["probe_gain"],
            "ov_probe_gain": over["probe_gain"],

            "un_early_mean": under["early_mean"],
            "ov_early_mean": over["early_mean"],

            "un_early_max": under["early_max"],
            "ov_early_max": over["early_max"],

            "un_early_gain": under["early_gain"],
            "ov_early_gain": over["early_gain"],

            "un_full_max": under["full_max"],
            "ov_full_max": over["full_max"],

            "un_epochs_used": under["epochs_used"],
            "ov_epochs_used": over["epochs_used"],
        }
        self._qa_end()
        return None

    def get_ternary_results(self, **kwargs):
        """
        Keep Apollo ternary logic, but enrich score construction with
        early-trajectory and probe-based signals.
        """
        print("Calculating IRIS_v3 Ternary Results!")

        gt, under_conf, over_conf, under_pred, over_pred = {}, {}, {}, {}, {}
        under_probe, over_probe = {}, {}
        under_early_gain, over_early_gain = {}, {}

        for name in ["unlearn", "retain", "test"]:
            gt[name] = torch.cat([self.summary[name][i]["target_label"] for i in self.summary[name]], dim=0).cpu().numpy()

            under_conf[name] = np.array([self.summary[name][i]["un_conf"] for i in self.summary[name]], dtype=object)
            over_conf[name] = np.array([self.summary[name][i]["ov_conf"] for i in self.summary[name]], dtype=object)

            under_pred[name] = np.array([self.summary[name][i]["un_pred"] for i in self.summary[name]], dtype=object)
            over_pred[name] = np.array([self.summary[name][i]["ov_pred"] for i in self.summary[name]], dtype=object)

            under_probe[name] = np.array([self.summary[name][i]["un_probe_gain"] for i in self.summary[name]])
            over_probe[name] = np.array([self.summary[name][i]["ov_probe_gain"] for i in self.summary[name]])

            under_early_gain[name] = np.array([self.summary[name][i]["un_early_gain"] for i in self.summary[name]])
            over_early_gain[name] = np.array([self.summary[name][i]["ov_early_gain"] for i in self.summary[name]])

        all_under_vals = []
        all_over_vals = []

        for name in ["unlearn", "retain", "test"]:
            for i in range(len(under_conf[name])):
                all_under_vals.extend(list(under_conf[name][i]))
                all_over_vals.extend(list(over_conf[name][i]))

        under_ths = np.percentile(all_under_vals, np.linspace(0, 100, 21))
        over_ths = np.percentile(all_over_vals, np.linspace(0, 100, 21))

        ternary_points = []
        threshold_pairs = []
        tpr_results = []
        accuracy_results = []
        full_classifications = []

        for under_th in tqdm(under_ths):
            for over_th in over_ths:
                classifications = {"unlearn": 0, "retain": 0, "test": 0}
                ground_truth_counts = {"unlearn": 0, "retain": 0, "test": 0}
                correct_classifications = {"unlearn": 0, "retain": 0, "test": 0}
                total_samples = 0

                for name in ["unlearn", "retain", "test"]:
                    for i in range(len(under_conf[name])):
                        true_label = gt[name][i]
                        ground_truth_counts[name] += 1

                        under_scores = []
                        over_scores = []

                        under_len = len(under_conf[name][i])
                        over_len = len(over_conf[name][i])
                        under_pred_len = len(under_pred[name][i])
                        over_pred_len = len(over_pred[name][i])

                        common_epochs = min(under_len, over_len, under_pred_len, over_pred_len)

                        if common_epochs == 0:
                            if name == "retain":
                                predicted_class = "retain"
                                classifications["retain"] += 1
                            elif name == "test":
                                predicted_class = "test"
                                classifications["test"] += 1
                            else:
                                predicted_class = "unlearn"
                                classifications["unlearn"] += 1

                            if predicted_class == name:
                                correct_classifications[name] += 1

                            total_samples += 1
                            continue

                        for epoch in range(common_epochs):
                            under_score = under_conf[name][i][epoch] - under_th
                            over_score = over_th - over_conf[name][i][epoch]

                            under_correct = (under_pred[name][i][epoch] == true_label)
                            over_correct = (over_pred[name][i][epoch] == true_label)

                            if not under_correct:
                                under_score -= 1.0
                            if not over_correct:
                                over_score += 1.0

                            under_scores.append(under_score)
                            over_scores.append(over_score)

                        # Add IRIS_v3 extras
                        if self.iris_use_early_features:
                            under_scores = [
                                s + 0.25 * under_probe[name][i] + 0.25 * under_early_gain[name][i]
                                for s in under_scores
                            ]
                            over_scores = [
                                s + 0.25 * over_probe[name][i] + 0.25 * over_early_gain[name][i]
                                for s in over_scores
                            ]

                        best_under_epoch = int(np.argmax(under_scores))
                        best_over_epoch = int(np.argmax(over_scores))

                        max_under_score = under_scores[best_under_epoch]
                        max_over_score = over_scores[best_over_epoch]

                        under_success = max_under_score > 0
                        over_success = max_over_score > 0

                        if under_success and over_success:
                            primary_under = max_under_score > max_over_score
                        elif under_success:
                            primary_under = True
                        elif over_success:
                            primary_under = False
                        else:
                            if name == "retain":
                                predicted_class = "retain"
                                classifications["retain"] += 1
                            elif name == "test":
                                predicted_class = "test"
                                classifications["test"] += 1
                            else:
                                predicted_class = "unlearn"
                                classifications["unlearn"] += 1

                            if predicted_class == name:
                                correct_classifications[name] += 1

                            total_samples += 1
                            continue

                        if primary_under:
                            epoch = best_under_epoch
                            is_correct = (under_pred[name][i][epoch] == true_label)

                            if is_correct and under_conf[name][i][epoch] > under_th:
                                predicted_class = "retain"
                                classifications["retain"] += 1
                            else:
                                predicted_class = "unlearn"
                                classifications["unlearn"] += 1
                        else:
                            epoch = best_over_epoch
                            is_correct = (over_pred[name][i][epoch] == true_label)

                            if (not is_correct) or (over_conf[name][i][epoch] < over_th):
                                predicted_class = "test"
                                classifications["test"] += 1
                            else:
                                predicted_class = "unlearn"
                                classifications["unlearn"] += 1

                        if predicted_class == name:
                            correct_classifications[name] += 1

                        total_samples += 1

                if total_samples > 0:
                    ternary_point = [
                        classifications["unlearn"] / total_samples,
                        classifications["retain"] / total_samples,
                        classifications["test"] / total_samples,
                    ]
                    ternary_points.append(ternary_point)
                    threshold_pairs.append((under_th, over_th))

                    tpr = {
                        "unlearn": correct_classifications["unlearn"] / ground_truth_counts["unlearn"] if ground_truth_counts["unlearn"] > 0 else 0.0,
                        "retain": correct_classifications["retain"] / ground_truth_counts["retain"] if ground_truth_counts["retain"] > 0 else 0.0,
                        "test": correct_classifications["test"] / ground_truth_counts["test"] if ground_truth_counts["test"] > 0 else 0.0,
                    }
                    tpr_results.append(tpr)

                    total_correct = sum(correct_classifications.values())
                    overall_accuracy = total_correct / total_samples
                    accuracy_results.append(overall_accuracy)

                    full_classifications.append({
                        "classifications": classifications.copy(),
                        "ground_truth_counts": ground_truth_counts.copy(),
                        "correct_classifications": correct_classifications.copy(),
                        "total_samples": total_samples,
                        "tpr": tpr.copy(),
                        "accuracy": overall_accuracy,
                    })

        return {
            "ternary_points": np.array(ternary_points),
            "threshold_data": np.array(threshold_pairs),
            "tpr_results": tpr_results,
            "accuracy_results": np.array(accuracy_results),
            "full_classifications": full_classifications,
        }