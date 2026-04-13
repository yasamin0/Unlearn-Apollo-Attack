from typing import Dict, Any, List

import numpy as np
import torch

from query_audit import QueryAudit

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IRISBinaryDirectional:
    """
    IRIS binary attack - disagreement based version

    Goal:
    - more Apollo-like than pure local instability
    - still lower-query than Apollo
    - binary task:
        positive = unlearn
        negative = retain + test

    Key idea:
    score a sample by how abnormal the target model's directional behavior is
    relative to the shadow ensemble.
    """

    def __init__(self, model, shadow_models, args, device=None):
        self.model = model
        self.shadow_models = list(shadow_models) if shadow_models is not None else []
        self.args = args
        self.device = device if device is not None else DEVICE

        self.num_directions = int(getattr(args, "iris_num_directions", 8))
        self.radius_min = float(getattr(args, "iris_radius_min", 0.02))
        self.radius_max = float(getattr(args, "iris_radius_max", 0.30))
        self.radius_steps = int(getattr(args, "iris_radius_steps", 8))
        self.decision_threshold = float(getattr(args, "iris_binary_threshold", 0.0))

        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.eps = 1e-8
        self.max_abs_z = 5.0

        self.model.eval()
        for m in self.shadow_models:
            m.eval()

        self.query_audit = QueryAudit()

    def _qa_start(self, group_name, sample_id):
        self.query_audit.start_sample(group_name, int(sample_id))

    def _qa_target(self, n=1):
        self.query_audit.add_target(n)

    def _qa_shadow(self, n=1):
        if hasattr(self.query_audit, "add_shadow"):
            self.query_audit.add_shadow(n)
        elif hasattr(self.query_audit, "add_shadow_forward"):
            self.query_audit.add_shadow_forward(n)
        elif hasattr(self.query_audit, "add_shadow_forwards"):
            self.query_audit.add_shadow_forwards(n)

    def _qa_end(self):
        self.query_audit.end_sample()

    def get_query_audit_summary(self):
        return self.query_audit.summary()

    @torch.no_grad()
    def get_hard_label(self, model, x: torch.Tensor, is_target: bool = True) -> int:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        if is_target:
            self._qa_target(1)
        else:
            self._qa_shadow(1)

        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
        return int(pred)

    def build_radius_schedule(self) -> List[float]:
        if self.radius_steps <= 1:
            return [self.radius_max]
        return np.linspace(self.radius_min, self.radius_max, self.radius_steps).tolist()

    def random_unit_direction(self, x: torch.Tensor) -> torch.Tensor:
        d = torch.randn_like(x)
        norm = d.view(-1).norm(p=2).item()
        if norm < 1e-12:
            return self.random_unit_direction(x)
        return d / norm

    def _extract_path_stats_from_labels(self, labels_along_path: List[int], clean_label: int, radius_schedule: List[float]) -> Dict[str, Any]:
        flip_positions = []
        alt_labels = []

        for i, pred in enumerate(labels_along_path):
            if pred != clean_label:
                flip_positions.append(i)
                alt_labels.append(int(pred))

        no_flip = len(flip_positions) == 0

        if no_flip:
            first_flip_idx = None
            first_flip_radius = None
            persistence = 0.0
            num_flips = 0
            flip_count_ratio = 0.0
            dominant_alt_label = None
            dominant_alt_ratio = 0.0
            transition_count = 0
            oscillation_score = 0.0
            stable_after_first = 0.0
        else:
            first_flip_idx = int(flip_positions[0])
            first_flip_radius = float(radius_schedule[first_flip_idx])

            suffix = labels_along_path[first_flip_idx:]
            changed_flags = [int(v != clean_label) for v in suffix]
            persistence = float(sum(changed_flags) / len(changed_flags))

            num_flips = int(sum(v != clean_label for v in labels_along_path))
            flip_count_ratio = float(num_flips / max(len(labels_along_path), 1))

            vals, counts = np.unique(np.array(alt_labels), return_counts=True)
            dominant_alt_label = int(vals[np.argmax(counts)])
            dominant_alt_ratio = float(np.max(counts) / max(len(alt_labels), 1))

            transition_count = 0
            for a, b in zip(labels_along_path[:-1], labels_along_path[1:]):
                if a != b:
                    transition_count += 1
            oscillation_score = float(transition_count / max(len(labels_along_path) - 1, 1))

            stable_after_first = 1.0
            for v in suffix:
                if v == clean_label:
                    stable_after_first = 0.0
                    break

        return {
            "labels_along_path": labels_along_path,
            "no_flip": bool(no_flip),
            "first_flip_idx": first_flip_idx,
            "first_flip_radius": first_flip_radius,
            "persistence_after_flip": float(persistence),
            "num_flips": int(num_flips),
            "flip_count_ratio": float(flip_count_ratio),
            "dominant_alt_label": dominant_alt_label,
            "dominant_alt_ratio": float(dominant_alt_ratio),
            "transition_count": int(transition_count),
            "oscillation_score": float(oscillation_score),
            "stable_after_first": float(stable_after_first),
        }

    def _probe_target_and_shadows_single_direction(
        self,
        x: torch.Tensor,
        direction: torch.Tensor,
        radius_schedule: List[float],
    ) -> Dict[str, Any]:
        target_clean = self.get_hard_label(self.model, x, is_target=True)

        shadow_clean_labels = []
        for sm in self.shadow_models:
            shadow_clean_labels.append(self.get_hard_label(sm, x, is_target=False))

        target_labels_path = []
        shadow_labels_paths = [[] for _ in self.shadow_models]

        for r in radius_schedule:
            z = x + r * direction
            z = torch.clamp(z, min=self.clamp_min, max=self.clamp_max)

            target_pred = self.get_hard_label(self.model, z, is_target=True)
            target_labels_path.append(int(target_pred))

            for j, sm in enumerate(self.shadow_models):
                shadow_pred = self.get_hard_label(sm, z, is_target=False)
                shadow_labels_paths[j].append(int(shadow_pred))

        target_stats = self._extract_path_stats_from_labels(
            labels_along_path=target_labels_path,
            clean_label=target_clean,
            radius_schedule=radius_schedule,
        )

        shadow_stats = []
        for j in range(len(self.shadow_models)):
            shadow_stats.append(
                self._extract_path_stats_from_labels(
                    labels_along_path=shadow_labels_paths[j],
                    clean_label=shadow_clean_labels[j],
                    radius_schedule=radius_schedule,
                )
            )

        disagreement_path = []
        alt_disagreement_path = []

        for i in range(len(radius_schedule)):
            shadow_preds_i = [shadow_labels_paths[j][i] for j in range(len(self.shadow_models))]
            if len(shadow_preds_i) == 0:
                disagreement_path.append(0.0)
                alt_disagreement_path.append(0.0)
                continue

            disagree = np.mean([int(p != target_labels_path[i]) for p in shadow_preds_i])
            disagreement_path.append(float(disagree))

            target_changed = int(target_labels_path[i] != target_clean)
            if target_changed == 0:
                alt_disagreement_path.append(0.0)
            else:
                alt_dis = np.mean([int(p != target_labels_path[i]) for p in shadow_preds_i])
                alt_disagreement_path.append(float(alt_dis))

        # compare target stats with shadow stats
        shadow_first_flip = []
        shadow_persistence = []
        shadow_flip_density = []
        shadow_alt_consistency = []
        shadow_stability_after_first = []
        shadow_oscillation = []

        for s in shadow_stats:
            ff = s["first_flip_radius"] if s["first_flip_radius"] is not None else self.radius_max
            shadow_first_flip.append(float(ff))
            shadow_persistence.append(float(s["persistence_after_flip"]))
            shadow_flip_density.append(float(s["flip_count_ratio"]))
            shadow_alt_consistency.append(float(s["dominant_alt_ratio"]))
            shadow_stability_after_first.append(float(s["stable_after_first"]))
            shadow_oscillation.append(float(s["oscillation_score"]))

        target_ff = target_stats["first_flip_radius"] if target_stats["first_flip_radius"] is not None else self.radius_max
        target_persistence = float(target_stats["persistence_after_flip"])
        target_flip_density = float(target_stats["flip_count_ratio"])
        target_alt_consistency = float(target_stats["dominant_alt_ratio"])
        target_stability_after_first = float(target_stats["stable_after_first"])
        target_oscillation = float(target_stats["oscillation_score"])

        return {
            "target_clean_label": int(target_clean),
            "shadow_clean_labels": [int(v) for v in shadow_clean_labels],
            "target_labels_path": target_labels_path,
            "shadow_labels_paths": shadow_labels_paths,
            "target_stats": target_stats,
            "shadow_stats": shadow_stats,
            "mean_disagreement": float(np.mean(disagreement_path)) if len(disagreement_path) > 0 else 0.0,
            "max_disagreement": float(np.max(disagreement_path)) if len(disagreement_path) > 0 else 0.0,
            "mean_alt_disagreement": float(np.mean(alt_disagreement_path)) if len(alt_disagreement_path) > 0 else 0.0,
            "target_minus_shadow_first_flip": float(target_ff - np.mean(shadow_first_flip)) if len(shadow_first_flip) > 0 else 0.0,
            "target_minus_shadow_persistence": float(target_persistence - np.mean(shadow_persistence)) if len(shadow_persistence) > 0 else 0.0,
            "target_minus_shadow_flip_density": float(target_flip_density - np.mean(shadow_flip_density)) if len(shadow_flip_density) > 0 else 0.0,
            "target_minus_shadow_alt_consistency": float(target_alt_consistency - np.mean(shadow_alt_consistency)) if len(shadow_alt_consistency) > 0 else 0.0,
            "target_minus_shadow_stability_after_first": float(target_stability_after_first - np.mean(shadow_stability_after_first)) if len(shadow_stability_after_first) > 0 else 0.0,
            "target_minus_shadow_oscillation": float(target_oscillation - np.mean(shadow_oscillation)) if len(shadow_oscillation) > 0 else 0.0,
            "target_first_flip_radius": float(target_ff),
            "shadow_mean_first_flip_radius": float(np.mean(shadow_first_flip)) if len(shadow_first_flip) > 0 else self.radius_max,
            "target_persistence": float(target_persistence),
            "shadow_mean_persistence": float(np.mean(shadow_persistence)) if len(shadow_persistence) > 0 else 0.0,
            "target_flip_density": float(target_flip_density),
            "shadow_mean_flip_density": float(np.mean(shadow_flip_density)) if len(shadow_flip_density) > 0 else 0.0,
        }

    def _z(self, value: float, values: List[float]) -> float:
        if len(values) == 0:
            return 0.0
        mu = float(np.mean(values))
        sigma = float(np.std(values))
        if sigma < self.eps:
            sigma = 1.0
        z = (float(value) - mu) / sigma
        return float(np.clip(z, -self.max_abs_z, self.max_abs_z))

    def score_sample(self, x: torch.Tensor, y: int) -> Dict[str, Any]:
        radius_schedule = self.build_radius_schedule()
        directions = [self.random_unit_direction(x) for _ in range(self.num_directions)]

        direction_results = []
        for d in directions:
            direction_results.append(
                self._probe_target_and_shadows_single_direction(
                    x=x,
                    direction=d,
                    radius_schedule=radius_schedule,
                )
            )

        mean_disagreement_vals = [r["mean_disagreement"] for r in direction_results]
        max_disagreement_vals = [r["max_disagreement"] for r in direction_results]
        mean_alt_disagreement_vals = [r["mean_alt_disagreement"] for r in direction_results]

        delta_ff_vals = [r["target_minus_shadow_first_flip"] for r in direction_results]
        delta_persistence_vals = [r["target_minus_shadow_persistence"] for r in direction_results]
        delta_flip_density_vals = [r["target_minus_shadow_flip_density"] for r in direction_results]
        delta_alt_consistency_vals = [r["target_minus_shadow_alt_consistency"] for r in direction_results]
        delta_stability_vals = [r["target_minus_shadow_stability_after_first"] for r in direction_results]
        delta_oscillation_vals = [r["target_minus_shadow_oscillation"] for r in direction_results]

        # raw aggregated features
        feat = {
            "mean_disagreement": float(np.mean(mean_disagreement_vals)),
            "max_disagreement": float(np.mean(max_disagreement_vals)),
            "mean_alt_disagreement": float(np.mean(mean_alt_disagreement_vals)),
            "mean_delta_first_flip": float(np.mean(delta_ff_vals)),
            "mean_delta_persistence": float(np.mean(delta_persistence_vals)),
            "mean_delta_flip_density": float(np.mean(delta_flip_density_vals)),
            "mean_delta_alt_consistency": float(np.mean(delta_alt_consistency_vals)),
            "mean_delta_stability_after_first": float(np.mean(delta_stability_vals)),
            "mean_delta_oscillation": float(np.mean(delta_oscillation_vals)),
        }

        # centered direction-wise abnormalities
        feat["z_mean_disagreement"] = self._z(feat["mean_disagreement"], mean_disagreement_vals)
        feat["z_max_disagreement"] = self._z(feat["max_disagreement"], max_disagreement_vals)
        feat["z_mean_alt_disagreement"] = self._z(feat["mean_alt_disagreement"], mean_alt_disagreement_vals)
        feat["z_mean_delta_first_flip"] = self._z(feat["mean_delta_first_flip"], delta_ff_vals)
        feat["z_mean_delta_persistence"] = self._z(feat["mean_delta_persistence"], delta_persistence_vals)
        feat["z_mean_delta_flip_density"] = self._z(feat["mean_delta_flip_density"], delta_flip_density_vals)
        feat["z_mean_delta_alt_consistency"] = self._z(feat["mean_delta_alt_consistency"], delta_alt_consistency_vals)
        feat["z_mean_delta_stability_after_first"] = self._z(feat["mean_delta_stability_after_first"], delta_stability_vals)
        feat["z_mean_delta_oscillation"] = self._z(feat["mean_delta_oscillation"], delta_oscillation_vals)

        # IMPORTANT:
        # target flipping EARLIER than shadows => delta_first_flip negative
        # so we use minus sign on that term
        iris_score = (
            0.28 * feat["mean_disagreement"] +
            0.18 * feat["max_disagreement"] +
            0.18 * feat["mean_alt_disagreement"] -
            0.16 * feat["mean_delta_first_flip"] +
            0.10 * feat["mean_delta_persistence"] +
            0.10 * feat["mean_delta_flip_density"] +
            0.08 * feat["mean_delta_alt_consistency"] +
            0.08 * feat["mean_delta_stability_after_first"] -
            0.06 * feat["mean_delta_oscillation"]
        )

        pred_binary = int(iris_score >= self.decision_threshold)

        out = {
            "target_label": int(y),
            "pred_clean": int(direction_results[0]["target_clean_label"]) if len(direction_results) > 0 else int(y),
            "radius_schedule": [float(r) for r in radius_schedule],
            "num_shadow_models_used": int(len(self.shadow_models)),
            "num_directions_used": int(self.num_directions),
            **feat,
            "iris_score": float(iris_score),
            "binary_pred": int(pred_binary),
        }

        return out

    def _unpack_batch(self, batch, fallback_start_id: int = 0):
        items = []

        if isinstance(batch, (list, tuple)):
            if len(batch) > 0 and isinstance(batch[0], (list, tuple)) and len(batch[0]) == 3:
                for sample in batch:
                    sid, x, y = sample
                    items.append((int(sid), x, int(y)))
                return items

            if len(batch) == 3:
                sample_ids, xs, ys = batch
                bs = len(xs)
                for i in range(bs):
                    items.append((int(sample_ids[i]), xs[i], int(ys[i])))
                return items

            if len(batch) == 2:
                xs, ys = batch
                bs = len(xs)
                for i in range(bs):
                    items.append((int(fallback_start_id + i), xs[i], int(ys[i])))
                return items

        raise ValueError("Unsupported batch format in IRISBinaryDirectional")

    def run_group(self, group_loader, group_ids, group_name: str) -> Dict[int, Dict[str, Any]]:
        group_results: Dict[int, Dict[str, Any]] = {}
        sample_counter = 0

        for batch in group_loader:
            items = self._unpack_batch(batch, fallback_start_id=sample_counter)
            for _, x, y in items:
                real_sample_id = int(group_ids[sample_counter])

                self._qa_start(group_name, real_sample_id)
                group_results[real_sample_id] = self.score_sample(x, y)
                self._qa_end()

                sample_counter += 1

        return group_results

    def run(self, target_groups: Dict[str, Any], target_idxs: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
        return {
            "unlearn": self.run_group(target_groups["unlearn"], target_idxs["unlearn"], "unlearn"),
            "retain": self.run_group(target_groups["retain"], target_idxs["retain"], "retain"),
            "test": self.run_group(target_groups["test"], target_idxs["test"], "test"),
        }