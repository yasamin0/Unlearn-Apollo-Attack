from typing import Dict, Any, List

import numpy as np
import torch

from query_audit import QueryAudit

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IRISBinaryDirectional:
    """
    Improved IRIS binary directional attack.

    Main changes:
    - shadow-assisted calibration
    - score centering with relative z-features
    - richer diagnostics
    - less biased score distribution around zero
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
        self.eps_std = 1e-6
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

    def probe_direction_for_model(
        self,
        model,
        x: torch.Tensor,
        clean_label: int,
        direction: torch.Tensor,
        radius_schedule: List[float],
        is_target: bool,
    ) -> Dict[str, Any]:
        labels_along_path = []
        flip_positions = []
        alt_labels = []

        for i, r in enumerate(radius_schedule):
            z = x + r * direction
            z = torch.clamp(z, min=self.clamp_min, max=self.clamp_max)
            pred = self.get_hard_label(model, z, is_target=is_target)
            labels_along_path.append(int(pred))

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

    def aggregate_directional_features(
        self,
        direction_results: List[Dict[str, Any]],
        radius_schedule: List[float],
    ) -> Dict[str, float]:
        max_radius = float(radius_schedule[-1])
        no_flip_flags = np.array([int(r["no_flip"]) for r in direction_results], dtype=float)
        flip_flags = 1.0 - no_flip_flags

        flip_fraction = float(np.mean(flip_flags))
        no_flip_fraction = float(np.mean(no_flip_flags))

        flip_radii = [r["first_flip_radius"] for r in direction_results if not r["no_flip"]]
        persistences = [r["persistence_after_flip"] for r in direction_results if not r["no_flip"]]
        flip_count_ratios = [r["flip_count_ratio"] for r in direction_results if not r["no_flip"]]
        dominant_alt_ratios = [r["dominant_alt_ratio"] for r in direction_results if not r["no_flip"]]
        oscillation_scores = [r["oscillation_score"] for r in direction_results if not r["no_flip"]]
        stable_after_first_vals = [r["stable_after_first"] for r in direction_results if not r["no_flip"]]

        mean_first_flip_radius = float(np.mean(flip_radii)) if len(flip_radii) > 0 else max_radius
        min_first_flip_radius = float(np.min(flip_radii)) if len(flip_radii) > 0 else max_radius
        std_first_flip_radius = float(np.std(flip_radii)) if len(flip_radii) > 1 else 0.0

        mean_persistence = float(np.mean(persistences)) if len(persistences) > 0 else 0.0
        mean_flip_count_ratio = float(np.mean(flip_count_ratios)) if len(flip_count_ratios) > 0 else 0.0
        mean_dominant_alt_ratio = float(np.mean(dominant_alt_ratios)) if len(dominant_alt_ratios) > 0 else 0.0
        mean_oscillation_score = float(np.mean(oscillation_scores)) if len(oscillation_scores) > 0 else 0.0
        mean_stable_after_first = float(np.mean(stable_after_first_vals)) if len(stable_after_first_vals) > 0 else 0.0

        first_flip_norm = float(mean_first_flip_radius / max_radius) if max_radius > 0 else 1.0
        min_flip_norm = float(min_first_flip_radius / max_radius) if max_radius > 0 else 1.0
        early_flip_score = float(1.0 - first_flip_norm) if max_radius > 0 else 0.0
        flip_spread_ratio = float(std_first_flip_radius / max_radius) if max_radius > 0 else 0.0

        return {
            "flip_fraction": flip_fraction,
            "no_flip_fraction": no_flip_fraction,
            "mean_first_flip_radius": mean_first_flip_radius,
            "min_first_flip_radius": min_first_flip_radius,
            "std_first_flip_radius": std_first_flip_radius,
            "first_flip_norm": first_flip_norm,
            "min_flip_norm": min_flip_norm,
            "mean_persistence": mean_persistence,
            "mean_flip_count_ratio": mean_flip_count_ratio,
            "mean_dominant_alt_ratio": mean_dominant_alt_ratio,
            "mean_oscillation_score": mean_oscillation_score,
            "mean_stable_after_first": mean_stable_after_first,
            "early_flip_score": early_flip_score,
            "flip_spread_ratio": flip_spread_ratio,
        }

    def _shadow_reference_stats(self, shadow_aggs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        keys = [
            "flip_fraction",
            "no_flip_fraction",
            "early_flip_score",
            "mean_persistence",
            "mean_flip_count_ratio",
            "mean_dominant_alt_ratio",
            "mean_oscillation_score",
            "mean_stable_after_first",
            "first_flip_norm",
            "min_flip_norm",
            "flip_spread_ratio",
        ]

        ref = {}
        for key in keys:
            vals = [float(a[key]) for a in shadow_aggs]
            ref[key] = {
                "mean": float(np.mean(vals)) if len(vals) > 0 else 0.0,
                "std": float(np.std(vals)) if len(vals) > 1 else self.eps_std,
            }
            if ref[key]["std"] < self.eps_std:
                ref[key]["std"] = self.eps_std
        return ref

    def _clip_z(self, z: float) -> float:
        return float(np.clip(z, -self.max_abs_z, self.max_abs_z))

    def build_feature_vector(self, target_agg: Dict[str, float], shadow_ref: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        def z(key):
            return self._clip_z((target_agg[key] - shadow_ref[key]["mean"]) / shadow_ref[key]["std"])

        features = {
            "raw_flip_fraction": float(target_agg["flip_fraction"]),
            "raw_no_flip_fraction": float(target_agg["no_flip_fraction"]),
            "raw_early_flip_score": float(target_agg["early_flip_score"]),
            "raw_persistence": float(target_agg["mean_persistence"]),
            "raw_flip_density": float(target_agg["mean_flip_count_ratio"]),
            "raw_alt_consistency": float(target_agg["mean_dominant_alt_ratio"]),
            "raw_oscillation": float(target_agg["mean_oscillation_score"]),
            "raw_stability_after_flip": float(target_agg["mean_stable_after_first"]),
            "raw_first_flip_norm": float(target_agg["first_flip_norm"]),
            "raw_min_flip_norm": float(target_agg["min_flip_norm"]),
            "raw_flip_spread": float(target_agg["flip_spread_ratio"]),

            "rel_flip_fraction_z": z("flip_fraction"),
            "rel_no_flip_fraction_z": z("no_flip_fraction"),
            "rel_early_flip_z": z("early_flip_score"),
            "rel_persistence_z": z("mean_persistence"),
            "rel_flip_density_z": z("mean_flip_count_ratio"),
            "rel_alt_consistency_z": z("mean_dominant_alt_ratio"),
            "rel_oscillation_z": z("mean_oscillation_score"),
            "rel_stability_after_flip_z": z("mean_stable_after_first"),
            "rel_first_flip_norm_z": z("first_flip_norm"),
            "rel_min_flip_norm_z": z("min_flip_norm"),
            "rel_flip_spread_z": z("flip_spread_ratio"),
        }
        return features

    def compute_binary_score(self, feat: Dict[str, float]) -> float:
        """
        Centered score.
        Positive score should more often indicate "more unlearn-like",
        but evaluator still checks both polarities automatically.
        """
        score = (
            0.30 * feat["rel_flip_fraction_z"] +
            0.22 * feat["rel_early_flip_z"] +
            0.18 * feat["rel_persistence_z"] +
            0.12 * feat["rel_stability_after_flip_z"] +
            0.10 * feat["rel_flip_density_z"] +
            0.06 * feat["rel_alt_consistency_z"] -
            0.10 * feat["rel_oscillation_z"] -
            0.08 * feat["rel_no_flip_fraction_z"] -
            0.14 * feat["rel_first_flip_norm_z"] -
            0.08 * feat["rel_min_flip_norm_z"] -
            0.04 * feat["rel_flip_spread_z"] +
            0.04 * feat["raw_flip_fraction"] -
            0.03 * feat["raw_no_flip_fraction"]
        )
        return float(score)

    def classify_binary(self, score: float) -> int:
        return int(score >= self.decision_threshold)

    def _run_directional_probe_set(self, model, x, is_target: bool) -> Dict[str, Any]:
        clean_label = self.get_hard_label(model, x, is_target=is_target)
        radius_schedule = self.build_radius_schedule()

        directions = [self.random_unit_direction(x) for _ in range(self.num_directions)]
        direction_results = []
        for d in directions:
            res = self.probe_direction_for_model(
                model=model,
                x=x,
                clean_label=clean_label,
                direction=d,
                radius_schedule=radius_schedule,
                is_target=is_target,
            )
            direction_results.append(res)

        agg = self.aggregate_directional_features(direction_results, radius_schedule)
        return {
            "clean_label": int(clean_label),
            "radius_schedule": [float(r) for r in radius_schedule],
            "direction_results": direction_results,
            "agg": agg,
        }

    def score_sample(self, x: torch.Tensor, y: int) -> Dict[str, Any]:
        target_probe = self._run_directional_probe_set(self.model, x, is_target=True)
        target_agg = target_probe["agg"]

        shadow_aggs = []
        shadow_clean_labels = []

        for sm in self.shadow_models:
            shadow_probe = self._run_directional_probe_set(sm, x, is_target=False)
            shadow_aggs.append(shadow_probe["agg"])
            shadow_clean_labels.append(int(shadow_probe["clean_label"]))

        if len(shadow_aggs) == 0:
            # fallback if no shadow models are available
            shadow_ref = {
                key: {"mean": float(target_agg[key]), "std": 1.0}
                for key in [
                    "flip_fraction",
                    "no_flip_fraction",
                    "early_flip_score",
                    "mean_persistence",
                    "mean_flip_count_ratio",
                    "mean_dominant_alt_ratio",
                    "mean_oscillation_score",
                    "mean_stable_after_first",
                    "first_flip_norm",
                    "min_flip_norm",
                    "flip_spread_ratio",
                ]
            }
        else:
            shadow_ref = self._shadow_reference_stats(shadow_aggs)

        feat = self.build_feature_vector(target_agg=target_agg, shadow_ref=shadow_ref)
        iris_score = self.compute_binary_score(feat)
        pred_binary = self.classify_binary(iris_score)

        out = {
            "target_label": int(y),
            "pred_clean": int(target_probe["clean_label"]),
            "radius_schedule": target_probe["radius_schedule"],
            "num_shadow_models_used": int(len(self.shadow_models)),
            "shadow_clean_label_mean": float(np.mean(shadow_clean_labels)) if len(shadow_clean_labels) > 0 else 0.0,
            "shadow_clean_label_std": float(np.std(shadow_clean_labels)) if len(shadow_clean_labels) > 0 else 0.0,
            **target_agg,
            **feat,
            "iris_score": float(iris_score),
            "binary_pred": int(pred_binary),
        }

        # add compact shadow reference diagnostics
        for k, stats in shadow_ref.items():
            out[f"shadow_mean__{k}"] = float(stats["mean"])
            out[f"shadow_std__{k}"] = float(stats["std"])

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