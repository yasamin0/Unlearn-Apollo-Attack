from typing import Dict, Any, List, Tuple

import numpy as np
import torch

from query_audit import QueryAudit

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IRISBinaryDirectional:
    """
    IRIS binary attack - adaptive low-query disagreement version

    Goals:
    - lower query cost than Apollo
    - no optimizer loop
    - stronger shadow-relative features than previous IRIS variants
    - optional adaptive refinement only for ambiguous samples
    """

    def __init__(self, model, shadow_models, args, device=None):
        self.model = model
        self.shadow_models = list(shadow_models) if shadow_models is not None else []
        self.args = args
        self.device = device if device is not None else DEVICE

        self.num_directions = int(getattr(args, "iris_num_directions", 6))
        self.radius_min = float(getattr(args, "iris_radius_min", 0.02))
        self.radius_max = float(getattr(args, "iris_radius_max", 0.24))
        self.radius_steps = int(getattr(args, "iris_radius_steps", 6))
        self.decision_threshold = float(getattr(args, "iris_binary_threshold", 0.0))

        self.use_adaptive_refine = bool(getattr(args, "iris_use_adaptive_refine", False))
        self.stage1_num_directions = int(getattr(args, "iris_stage1_num_directions", 4))
        self.stage1_radius_steps = int(getattr(args, "iris_stage1_radius_steps", 4))
        self.base_num_shadow = int(getattr(args, "iris_base_num_shadow", 2))
        self.refine_margin = float(getattr(args, "iris_refine_margin", 0.08))

        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.eps = 1e-8

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

    def build_radius_schedule(self, steps: int = None) -> List[float]:
        n_steps = int(self.radius_steps if steps is None else steps)
        if n_steps <= 1:
            return [self.radius_max]
        return np.linspace(self.radius_min, self.radius_max, n_steps).tolist()

    def random_unit_direction(self, x: torch.Tensor) -> torch.Tensor:
        d = torch.randn_like(x)
        norm = d.view(-1).norm(p=2).item()
        if norm < 1e-12:
            return self.random_unit_direction(x)
        return d / norm

    def _extract_path_stats_from_labels(
        self,
        labels_along_path: List[int],
        clean_label: int,
        radius_schedule: List[float],
    ) -> Dict[str, Any]:
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

    def _split_radius_bands(self, values: List[float]) -> Tuple[float, float, float]:
        if len(values) == 0:
            return 0.0, 0.0, 0.0
        n = len(values)
        a = max(1, n // 3)
        b = max(a + 1, (2 * n) // 3)
        early = float(np.mean(values[:a])) if a > 0 else 0.0
        mid = float(np.mean(values[a:b])) if b > a else early
        late = float(np.mean(values[b:])) if n > b else mid
        return early, mid, late

    def _probe_with_subset(
        self,
        x: torch.Tensor,
        direction: torch.Tensor,
        radius_schedule: List[float],
        shadow_subset: List[int],
    ) -> Dict[str, Any]:
        target_clean = self.get_hard_label(self.model, x, is_target=True)

        shadow_clean_labels = []
        for j in shadow_subset:
            shadow_clean_labels.append(self.get_hard_label(self.shadow_models[j], x, is_target=False))

        target_labels_path = []
        shadow_labels_paths = [[] for _ in shadow_subset]

        for r in radius_schedule:
            z = x + r * direction
            z = torch.clamp(z, min=self.clamp_min, max=self.clamp_max)

            target_pred = self.get_hard_label(self.model, z, is_target=True)
            target_labels_path.append(int(target_pred))

            for loc, j in enumerate(shadow_subset):
                shadow_pred = self.get_hard_label(self.shadow_models[j], z, is_target=False)
                shadow_labels_paths[loc].append(int(shadow_pred))

        target_stats = self._extract_path_stats_from_labels(
            labels_along_path=target_labels_path,
            clean_label=target_clean,
            radius_schedule=radius_schedule,
        )

        shadow_stats = []
        for loc in range(len(shadow_subset)):
            shadow_stats.append(
                self._extract_path_stats_from_labels(
                    labels_along_path=shadow_labels_paths[loc],
                    clean_label=shadow_clean_labels[loc],
                    radius_schedule=radius_schedule,
                )
            )

        disagreement_path = []
        alt_disagreement_path = []
        target_flip_while_shadow_majority_not = []
        target_alt_mismatch_vs_shadow_majority = []

        for i in range(len(radius_schedule)):
            shadow_preds_i = [shadow_labels_paths[loc][i] for loc in range(len(shadow_subset))]
            if len(shadow_preds_i) == 0:
                disagreement_path.append(0.0)
                alt_disagreement_path.append(0.0)
                target_flip_while_shadow_majority_not.append(0.0)
                target_alt_mismatch_vs_shadow_majority.append(0.0)
                continue

            disagree = np.mean([int(p != target_labels_path[i]) for p in shadow_preds_i])
            disagreement_path.append(float(disagree))

            shadow_majority_not_clean = np.mean([int(p != shadow_clean_labels[loc]) for loc, p in enumerate(shadow_preds_i)])
            target_changed = int(target_labels_path[i] != target_clean)

            if target_changed == 1 and shadow_majority_not_clean < 0.5:
                target_flip_while_shadow_majority_not.append(1.0)
            else:
                target_flip_while_shadow_majority_not.append(0.0)

            if target_changed == 0:
                alt_disagreement_path.append(0.0)
                target_alt_mismatch_vs_shadow_majority.append(0.0)
            else:
                alt_dis = np.mean([int(p != target_labels_path[i]) for p in shadow_preds_i])
                alt_disagreement_path.append(float(alt_dis))

                vals, counts = np.unique(np.array(shadow_preds_i), return_counts=True)
                shadow_majority_label = int(vals[np.argmax(counts)])
                target_alt_mismatch_vs_shadow_majority.append(float(int(target_labels_path[i] != shadow_majority_label)))

        target_ff = target_stats["first_flip_radius"] if target_stats["first_flip_radius"] is not None else self.radius_max
        shadow_ffs = []
        shadow_persistences = []
        shadow_flip_densities = []
        shadow_alt_consistencies = []
        shadow_oscillations = []
        shadow_stabilities = []

        for s in shadow_stats:
            shadow_ffs.append(float(s["first_flip_radius"] if s["first_flip_radius"] is not None else self.radius_max))
            shadow_persistences.append(float(s["persistence_after_flip"]))
            shadow_flip_densities.append(float(s["flip_count_ratio"]))
            shadow_alt_consistencies.append(float(s["dominant_alt_ratio"]))
            shadow_oscillations.append(float(s["oscillation_score"]))
            shadow_stabilities.append(float(s["stable_after_first"]))

        mean_shadow_ff = float(np.mean(shadow_ffs)) if len(shadow_ffs) > 0 else self.radius_max

        early_dis, mid_dis, late_dis = self._split_radius_bands(disagreement_path)
        early_alt_dis, mid_alt_dis, late_alt_dis = self._split_radius_bands(alt_disagreement_path)

        return {
            "target_clean_label": int(target_clean),
            "target_stats": target_stats,
            "mean_disagreement": float(np.mean(disagreement_path)) if len(disagreement_path) > 0 else 0.0,
            "max_disagreement": float(np.max(disagreement_path)) if len(disagreement_path) > 0 else 0.0,
            "early_disagreement": early_dis,
            "mid_disagreement": mid_dis,
            "late_disagreement": late_dis,
            "mean_alt_disagreement": float(np.mean(alt_disagreement_path)) if len(alt_disagreement_path) > 0 else 0.0,
            "early_alt_disagreement": early_alt_dis,
            "mid_alt_disagreement": mid_alt_dis,
            "late_alt_disagreement": late_alt_dis,
            "target_flip_while_shadow_majority_not": float(np.mean(target_flip_while_shadow_majority_not)) if len(target_flip_while_shadow_majority_not) > 0 else 0.0,
            "target_alt_mismatch_vs_shadow_majority": float(np.mean(target_alt_mismatch_vs_shadow_majority)) if len(target_alt_mismatch_vs_shadow_majority) > 0 else 0.0,
            "target_minus_shadow_first_flip": float(target_ff - mean_shadow_ff),
            "target_minus_shadow_persistence": float(target_stats["persistence_after_flip"] - (np.mean(shadow_persistences) if len(shadow_persistences) > 0 else 0.0)),
            "target_minus_shadow_flip_density": float(target_stats["flip_count_ratio"] - (np.mean(shadow_flip_densities) if len(shadow_flip_densities) > 0 else 0.0)),
            "target_minus_shadow_alt_consistency": float(target_stats["dominant_alt_ratio"] - (np.mean(shadow_alt_consistencies) if len(shadow_alt_consistencies) > 0 else 0.0)),
            "target_minus_shadow_oscillation": float(target_stats["oscillation_score"] - (np.mean(shadow_oscillations) if len(shadow_oscillations) > 0 else 0.0)),
            "target_minus_shadow_stability_after_first": float(target_stats["stable_after_first"] - (np.mean(shadow_stabilities) if len(shadow_stabilities) > 0 else 0.0)),
        }

    def _aggregate_direction_results(self, direction_results: List[Dict[str, Any]]) -> Dict[str, float]:
        if len(direction_results) == 0:
            return {}

        def mean_of(key):
            return float(np.mean([float(r[key]) for r in direction_results]))

        def frac_of(predicate):
            return float(np.mean([1.0 if predicate(r) else 0.0 for r in direction_results]))

        feat = {
            "mean_disagreement": mean_of("mean_disagreement"),
            "max_disagreement": mean_of("max_disagreement"),
            "early_disagreement": mean_of("early_disagreement"),
            "mid_disagreement": mean_of("mid_disagreement"),
            "late_disagreement": mean_of("late_disagreement"),

            "mean_alt_disagreement": mean_of("mean_alt_disagreement"),
            "early_alt_disagreement": mean_of("early_alt_disagreement"),
            "mid_alt_disagreement": mean_of("mid_alt_disagreement"),
            "late_alt_disagreement": mean_of("late_alt_disagreement"),

            "target_flip_while_shadow_majority_not": mean_of("target_flip_while_shadow_majority_not"),
            "target_alt_mismatch_vs_shadow_majority": mean_of("target_alt_mismatch_vs_shadow_majority"),

            "mean_delta_first_flip": mean_of("target_minus_shadow_first_flip"),
            "mean_delta_persistence": mean_of("target_minus_shadow_persistence"),
            "mean_delta_flip_density": mean_of("target_minus_shadow_flip_density"),
            "mean_delta_alt_consistency": mean_of("target_minus_shadow_alt_consistency"),
            "mean_delta_oscillation": mean_of("target_minus_shadow_oscillation"),
            "mean_delta_stability_after_first": mean_of("target_minus_shadow_stability_after_first"),

            "frac_target_earlier_than_shadow": frac_of(lambda r: float(r["target_minus_shadow_first_flip"]) < 0.0),
            "frac_target_more_persistent_than_shadow": frac_of(lambda r: float(r["target_minus_shadow_persistence"]) > 0.0),
            "frac_target_denser_than_shadow": frac_of(lambda r: float(r["target_minus_shadow_flip_density"]) > 0.0),
            "frac_target_more_alt_mismatch_than_shadow": frac_of(lambda r: float(r["target_alt_mismatch_vs_shadow_majority"]) > 0.0),
        }

        return feat

    def _compute_score(self, feat: Dict[str, float]) -> float:
        # Important sign:
        # earlier target flip than shadow => mean_delta_first_flip negative => use minus sign
        score = (
            0.18 * feat["early_disagreement"] +
            0.10 * feat["mid_disagreement"] +
            0.04 * feat["late_disagreement"] +
            0.12 * feat["early_alt_disagreement"] +
            0.08 * feat["mid_alt_disagreement"] +
            0.05 * feat["target_flip_while_shadow_majority_not"] +
            0.06 * feat["target_alt_mismatch_vs_shadow_majority"] -
            0.12 * feat["mean_delta_first_flip"] +
            0.08 * feat["mean_delta_persistence"] +
            0.07 * feat["mean_delta_flip_density"] +
            0.05 * feat["mean_delta_alt_consistency"] +
            0.04 * feat["mean_delta_stability_after_first"] -
            0.04 * feat["mean_delta_oscillation"] +
            0.05 * feat["frac_target_earlier_than_shadow"] +
            0.03 * feat["frac_target_more_persistent_than_shadow"] +
            0.02 * feat["frac_target_denser_than_shadow"] +
            0.03 * feat["frac_target_more_alt_mismatch_than_shadow"]
        )
        return float(score)

    def _run_probe_block(
        self,
        x: torch.Tensor,
        num_directions: int,
        radius_steps: int,
        shadow_subset: List[int],
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        radius_schedule = self.build_radius_schedule(steps=radius_steps)
        direction_results = []

        for _ in range(num_directions):
            d = self.random_unit_direction(x)
            direction_results.append(
                self._probe_with_subset(
                    x=x,
                    direction=d,
                    radius_schedule=radius_schedule,
                    shadow_subset=shadow_subset,
                )
            )

        return direction_results, radius_schedule

    def score_sample(self, x: torch.Tensor, y: int) -> Dict[str, Any]:
        num_shadows_total = len(self.shadow_models)
        base_num_shadow = min(self.base_num_shadow, num_shadows_total)
        full_shadow_subset = list(range(num_shadows_total))
        stage1_shadow_subset = list(range(base_num_shadow))

        # Stage 1: cheap probe
        stage1_results, stage1_radius_schedule = self._run_probe_block(
            x=x,
            num_directions=min(self.stage1_num_directions, self.num_directions),
            radius_steps=min(self.stage1_radius_steps, self.radius_steps),
            shadow_subset=stage1_shadow_subset,
        )

        stage1_feat = self._aggregate_direction_results(stage1_results)
        stage1_score = self._compute_score(stage1_feat)

        refined = False
        final_results = list(stage1_results)
        final_radius_schedule = list(stage1_radius_schedule)
        final_shadow_subset = list(stage1_shadow_subset)

        # Optional adaptive refinement only for ambiguous samples
        if self.use_adaptive_refine and abs(stage1_score) < self.refine_margin:
            refined = True

            extra_dirs = max(self.num_directions - len(stage1_results), 0)
            if extra_dirs > 0:
                refine_results, refine_radius_schedule = self._run_probe_block(
                    x=x,
                    num_directions=extra_dirs,
                    radius_steps=self.radius_steps,
                    shadow_subset=full_shadow_subset,
                )
                final_results.extend(refine_results)
                final_radius_schedule = list(refine_radius_schedule)
                final_shadow_subset = list(full_shadow_subset)

        final_feat = self._aggregate_direction_results(final_results)
        iris_score = self._compute_score(final_feat)
        pred_binary = int(iris_score >= self.decision_threshold)

        out = {
            "target_label": int(y),
            "pred_clean": int(stage1_results[0]["target_clean_label"]) if len(stage1_results) > 0 else int(y),
            "radius_schedule": [float(r) for r in final_radius_schedule],
            "num_shadow_models_used": int(len(final_shadow_subset)),
            "num_directions_used": int(len(final_results)),
            "used_adaptive_refine": int(refined),
            "stage1_score": float(stage1_score),
            **final_feat,
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