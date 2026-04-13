from typing import Dict, Any, List, Tuple
import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier

from query_audit import QueryAudit

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IRISBinaryDirectional:
    """
    Learned IRIS:
    - low-query adaptive probing at target-time
    - offline attack-train set from shadow side
    - learned attack head (Random Forest)
    - no optimizer loop in attack-time
    """

    def __init__(self, model, shadow_models, dataset, shadow_col, target_split_orig, args, device=None):
        self.model = model
        self.shadow_models = list(shadow_models) if shadow_models is not None else []
        self.dataset = dataset
        self.shadow_col = shadow_col
        self.target_split_orig = target_split_orig
        self.args = args
        self.device = device if device is not None else DEVICE

        self.num_directions = int(getattr(args, "iris_num_directions", 6))
        self.radius_min = float(getattr(args, "iris_radius_min", 0.02))
        self.radius_max = float(getattr(args, "iris_radius_max", 0.24))
        self.radius_steps = int(getattr(args, "iris_radius_steps", 6))
        self.decision_threshold = float(getattr(args, "iris_binary_threshold", 0.5))

        self.use_adaptive_refine = bool(getattr(args, "iris_use_adaptive_refine", False))
        self.stage1_num_directions = int(getattr(args, "iris_stage1_num_directions", 4))
        self.stage1_radius_steps = int(getattr(args, "iris_stage1_radius_steps", 4))
        self.base_num_shadow = int(getattr(args, "iris_base_num_shadow", 2))
        self.refine_margin = float(getattr(args, "iris_refine_margin", 0.08))

        self.shadow_train_pos_per_model = int(getattr(args, "iris_shadow_train_pos_per_model", 30))
        self.shadow_train_neg_per_model = int(getattr(args, "iris_shadow_train_neg_per_model", 60))
        self.attack_head_seed = int(getattr(args, "iris_attack_head_seed", 42))

        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.eps = 1e-8

        self.model.eval()
        for m in self.shadow_models:
            m.eval()

        self.query_audit = QueryAudit()
        self.rng = np.random.RandomState(self.attack_head_seed)

        self.attack_head = None
        self.attack_feature_keys = None
        self.attack_head_meta = {}

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

        # NOTE:
        # during offline shadow-train feature extraction, _current is None,
        # so QueryAudit will not count anything.
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

    def _get_dataset_item_by_id(self, sample_id: int):
        sid = int(sample_id)

        try:
            if hasattr(self.dataset, "get_subset"):
                subset = self.dataset.get_subset([sid])
                item = subset[0]
            else:
                item = self.dataset[sid]
        except Exception:
            item = self.dataset[sid]

        if isinstance(item, (list, tuple)):
            if len(item) == 3:
                _, x, y = item
            elif len(item) >= 2:
                x, y = item[0], item[1]
            else:
                raise ValueError("Dataset item format unsupported.")
        else:
            raise ValueError("Dataset item format unsupported.")

        return x, int(y)

    def _probe_with_reference_models(
        self,
        target_model,
        reference_models: List,
        x: torch.Tensor,
        direction: torch.Tensor,
        radius_schedule: List[float],
    ) -> Dict[str, Any]:
        target_clean = self.get_hard_label(target_model, x, is_target=True)

        ref_clean_labels = []
        for rm in reference_models:
            ref_clean_labels.append(self.get_hard_label(rm, x, is_target=False))

        target_labels_path = []
        ref_labels_paths = [[] for _ in reference_models]

        for r in radius_schedule:
            z = x + r * direction
            z = torch.clamp(z, min=self.clamp_min, max=self.clamp_max)

            target_pred = self.get_hard_label(target_model, z, is_target=True)
            target_labels_path.append(int(target_pred))

            for loc, rm in enumerate(reference_models):
                ref_pred = self.get_hard_label(rm, z, is_target=False)
                ref_labels_paths[loc].append(int(ref_pred))

        target_stats = self._extract_path_stats_from_labels(
            labels_along_path=target_labels_path,
            clean_label=target_clean,
            radius_schedule=radius_schedule,
        )

        ref_stats = []
        for loc in range(len(reference_models)):
            ref_stats.append(
                self._extract_path_stats_from_labels(
                    labels_along_path=ref_labels_paths[loc],
                    clean_label=ref_clean_labels[loc],
                    radius_schedule=radius_schedule,
                )
            )

        disagreement_path = []
        alt_disagreement_path = []
        target_flip_while_ref_majority_not = []
        target_alt_mismatch_vs_ref_majority = []

        for i in range(len(radius_schedule)):
            ref_preds_i = [ref_labels_paths[loc][i] for loc in range(len(reference_models))]
            if len(ref_preds_i) == 0:
                disagreement_path.append(0.0)
                alt_disagreement_path.append(0.0)
                target_flip_while_ref_majority_not.append(0.0)
                target_alt_mismatch_vs_ref_majority.append(0.0)
                continue

            disagree = np.mean([int(p != target_labels_path[i]) for p in ref_preds_i])
            disagreement_path.append(float(disagree))

            ref_majority_not_clean = np.mean([int(p != ref_clean_labels[loc]) for loc, p in enumerate(ref_preds_i)])
            target_changed = int(target_labels_path[i] != target_clean)

            if target_changed == 1 and ref_majority_not_clean < 0.5:
                target_flip_while_ref_majority_not.append(1.0)
            else:
                target_flip_while_ref_majority_not.append(0.0)

            if target_changed == 0:
                alt_disagreement_path.append(0.0)
                target_alt_mismatch_vs_ref_majority.append(0.0)
            else:
                alt_dis = np.mean([int(p != target_labels_path[i]) for p in ref_preds_i])
                alt_disagreement_path.append(float(alt_dis))

                vals, counts = np.unique(np.array(ref_preds_i), return_counts=True)
                ref_majority_label = int(vals[np.argmax(counts)])
                target_alt_mismatch_vs_ref_majority.append(float(int(target_labels_path[i] != ref_majority_label)))

        target_ff = target_stats["first_flip_radius"] if target_stats["first_flip_radius"] is not None else self.radius_max

        ref_ffs = []
        ref_persistences = []
        ref_flip_densities = []
        ref_alt_consistencies = []
        ref_oscillations = []
        ref_stabilities = []

        for s in ref_stats:
            ref_ffs.append(float(s["first_flip_radius"] if s["first_flip_radius"] is not None else self.radius_max))
            ref_persistences.append(float(s["persistence_after_flip"]))
            ref_flip_densities.append(float(s["flip_count_ratio"]))
            ref_alt_consistencies.append(float(s["dominant_alt_ratio"]))
            ref_oscillations.append(float(s["oscillation_score"]))
            ref_stabilities.append(float(s["stable_after_first"]))

        mean_ref_ff = float(np.mean(ref_ffs)) if len(ref_ffs) > 0 else self.radius_max

        early_dis, mid_dis, late_dis = self._split_radius_bands(disagreement_path)
        early_alt_dis, mid_alt_dis, late_alt_dis = self._split_radius_bands(alt_disagreement_path)

        per_radius_feats = {}
        for ridx, val in enumerate(disagreement_path):
            per_radius_feats[f"disagreement_r{ridx}"] = float(val)

        for ridx, val in enumerate(alt_disagreement_path):
            per_radius_feats[f"alt_disagreement_r{ridx}"] = float(val)
        return {
            "target_clean_label": int(target_clean),
            "mean_disagreement": float(np.mean(disagreement_path)) if len(disagreement_path) > 0 else 0.0,
            "max_disagreement": float(np.max(disagreement_path)) if len(disagreement_path) > 0 else 0.0,
            "early_disagreement": early_dis,
            "mid_disagreement": mid_dis,
            "late_disagreement": late_dis,

            "mean_alt_disagreement": float(np.mean(alt_disagreement_path)) if len(alt_disagreement_path) > 0 else 0.0,
            "early_alt_disagreement": early_alt_dis,
            "mid_alt_disagreement": mid_alt_dis,
            "late_alt_disagreement": late_alt_dis,

            "target_flip_while_shadow_majority_not": float(np.mean(target_flip_while_ref_majority_not)) if len(target_flip_while_ref_majority_not) > 0 else 0.0,
            "target_alt_mismatch_vs_shadow_majority": float(np.mean(target_alt_mismatch_vs_ref_majority)) if len(target_alt_mismatch_vs_ref_majority) > 0 else 0.0,

            "target_minus_shadow_first_flip": float(target_ff - mean_ref_ff),
            "target_minus_shadow_persistence": float(target_stats["persistence_after_flip"] - (np.mean(ref_persistences) if len(ref_persistences) > 0 else 0.0)),
            "target_minus_shadow_flip_density": float(target_stats["flip_count_ratio"] - (np.mean(ref_flip_densities) if len(ref_flip_densities) > 0 else 0.0)),
            "target_minus_shadow_alt_consistency": float(target_stats["dominant_alt_ratio"] - (np.mean(ref_alt_consistencies) if len(ref_alt_consistencies) > 0 else 0.0)),
            "target_minus_shadow_oscillation": float(target_stats["oscillation_score"] - (np.mean(ref_oscillations) if len(ref_oscillations) > 0 else 0.0)),
            "target_minus_shadow_stability_after_first": float(target_stats["stable_after_first"] - (np.mean(ref_stabilities) if len(ref_stabilities) > 0 else 0.0)),

            **per_radius_feats,
        }

    def _aggregate_direction_results(self, direction_results: List[Dict[str, Any]]) -> Dict[str, float]:
        if len(direction_results) == 0:
            return {}

        def mean_of(key):
            return float(np.mean([float(r[key]) for r in direction_results]))

        def std_of(key):
            vals = [float(r[key]) for r in direction_results]
            return float(np.std(vals)) if len(vals) > 1 else 0.0

        def frac_of(predicate):
            return float(np.mean([1.0 if predicate(r) else 0.0 for r in direction_results]))
        per_radius_feat = {}

        sample_keys = direction_results[0].keys()
        disagreement_keys = sorted([k for k in sample_keys if k.startswith("disagreement_r")], key=lambda s: int(s.replace("disagreement_r", "")))
        alt_disagreement_keys = sorted([k for k in sample_keys if k.startswith("alt_disagreement_r")], key=lambda s: int(s.replace("alt_disagreement_r", "")))

        for k in disagreement_keys:
            per_radius_feat[f"mean_{k}"] = mean_of(k)
            per_radius_feat[f"std_{k}"] = std_of(k)

        for k in alt_disagreement_keys:
            per_radius_feat[f"mean_{k}"] = mean_of(k)
            per_radius_feat[f"std_{k}"] = std_of(k)

        feat = {
            "mean_disagreement": mean_of("mean_disagreement"),
            "std_disagreement": std_of("mean_disagreement"),
            "max_disagreement": mean_of("max_disagreement"),
            "early_disagreement": mean_of("early_disagreement"),
            "mid_disagreement": mean_of("mid_disagreement"),
            "late_disagreement": mean_of("late_disagreement"),

            "mean_alt_disagreement": mean_of("mean_alt_disagreement"),
            "std_alt_disagreement": std_of("mean_alt_disagreement"),
            "early_alt_disagreement": mean_of("early_alt_disagreement"),
            "mid_alt_disagreement": mean_of("mid_alt_disagreement"),
            "late_alt_disagreement": mean_of("late_alt_disagreement"),

            "target_flip_while_shadow_majority_not": mean_of("target_flip_while_shadow_majority_not"),
            "target_alt_mismatch_vs_shadow_majority": mean_of("target_alt_mismatch_vs_shadow_majority"),

            "mean_delta_first_flip": mean_of("target_minus_shadow_first_flip"),
            "std_delta_first_flip": std_of("target_minus_shadow_first_flip"),
            "mean_delta_persistence": mean_of("target_minus_shadow_persistence"),
            "mean_delta_flip_density": mean_of("target_minus_shadow_flip_density"),
            "mean_delta_alt_consistency": mean_of("target_minus_shadow_alt_consistency"),
            "mean_delta_oscillation": mean_of("target_minus_shadow_oscillation"),
            "mean_delta_stability_after_first": mean_of("target_minus_shadow_stability_after_first"),

            "frac_target_earlier_than_shadow": frac_of(lambda r: float(r["target_minus_shadow_first_flip"]) < 0.0),
            "frac_target_more_persistent_than_shadow": frac_of(lambda r: float(r["target_minus_shadow_persistence"]) > 0.0),
            "frac_target_denser_than_shadow": frac_of(lambda r: float(r["target_minus_shadow_flip_density"]) > 0.0),
            "frac_target_more_alt_mismatch_than_shadow": frac_of(lambda r: float(r["target_alt_mismatch_vs_shadow_majority"]) > 0.0),

            **per_radius_feat,
        }
        return feat

    def _legacy_score(self, feat: Dict[str, float]) -> float:
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
        target_model,
        reference_models: List,
        x: torch.Tensor,
        num_directions: int,
        radius_steps: int,
        max_ref_models: int,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        radius_schedule = self.build_radius_schedule(steps=radius_steps)
        direction_results = []

        ref_models = reference_models[:max_ref_models]

        for _ in range(num_directions):
            d = self.random_unit_direction(x)
            direction_results.append(
                self._probe_with_reference_models(
                    target_model=target_model,
                    reference_models=ref_models,
                    x=x,
                    direction=d,
                    radius_schedule=radius_schedule,
                )
            )

        return direction_results, radius_schedule

    def _extract_feature_dict_for_model(
        self,
        target_model,
        reference_models: List,
        x: torch.Tensor,
        y: int,
    ) -> Dict[str, Any]:
        num_refs_total = len(reference_models)
        base_num_ref = min(self.base_num_shadow, num_refs_total)
        full_ref_count = max(num_refs_total, 1)

        # Stage 1
        stage1_results, stage1_radius_schedule = self._run_probe_block(
            target_model=target_model,
            reference_models=reference_models,
            x=x,
            num_directions=min(self.stage1_num_directions, self.num_directions),
            radius_steps=min(self.stage1_radius_steps, self.radius_steps),
            max_ref_models=max(base_num_ref, 1),
        )

        stage1_feat = self._aggregate_direction_results(stage1_results)
        stage1_score = self._legacy_score(stage1_feat)

        refined = False
        final_results = list(stage1_results)
        final_radius_schedule = list(stage1_radius_schedule)
        final_num_refs_used = max(base_num_ref, 1)

        if self.use_adaptive_refine and abs(stage1_score) < self.refine_margin:
            refined = True
            extra_dirs = max(self.num_directions - len(stage1_results), 0)
            if extra_dirs > 0:
                refine_results, refine_radius_schedule = self._run_probe_block(
                    target_model=target_model,
                    reference_models=reference_models,
                    x=x,
                    num_directions=extra_dirs,
                    radius_steps=self.radius_steps,
                    max_ref_models=full_ref_count,
                )
                final_results.extend(refine_results)
                final_radius_schedule = list(refine_radius_schedule)
                final_num_refs_used = full_ref_count

        final_feat = self._aggregate_direction_results(final_results)
        legacy_score = self._legacy_score(final_feat)

        out = {
            "target_label": int(y),
            "pred_clean": int(stage1_results[0]["target_clean_label"]) if len(stage1_results) > 0 else int(y),
            "radius_schedule": [float(r) for r in final_radius_schedule],
            "num_shadow_models_used": int(final_num_refs_used),
            "num_directions_used": int(len(final_results)),
            "used_adaptive_refine": int(refined),
            "stage1_score": float(stage1_score),
            **final_feat,
            "legacy_score": float(legacy_score),
        }

        return out

    def _sample_ids(self, ids: List[int], n: int) -> List[int]:
        ids = list(sorted(set(int(v) for v in ids)))
        if len(ids) <= n:
            return ids
        sel = self.rng.choice(ids, size=n, replace=False)
        return [int(v) for v in sel.tolist()]


    def _build_attack_train_ids_from_target_splits(self):
        """
        Practical learned-head training set:
        positive = target unlearn pool
        negative = target retain pool + valid pool

        This avoids dependency on unknown shadow_col schema.
        """
        pos_pool = []
        neg_pool = []

        if isinstance(self.target_split_orig, dict):
            if "unlearn" in self.target_split_orig and self.target_split_orig["unlearn"] is not None:
                pos_pool = np.array(self.target_split_orig["unlearn"]).reshape(-1).astype(int).tolist()

            if "retain" in self.target_split_orig and self.target_split_orig["retain"] is not None:
                neg_pool.extend(np.array(self.target_split_orig["retain"]).reshape(-1).astype(int).tolist())

            if "valid" in self.target_split_orig and self.target_split_orig["valid"] is not None:
                neg_pool.extend(np.array(self.target_split_orig["valid"]).reshape(-1).astype(int).tolist())

        pos_pool = sorted(set(pos_pool))
        neg_pool = sorted(set(v for v in neg_pool if v not in set(pos_pool)))

        return pos_pool, neg_pool

    def _fit_attack_head_from_shadows(self):
        if self.attack_head is not None:
            return

        pos_pool, neg_pool = self._build_attack_train_ids_from_target_splits()

        pos_ids = self._sample_ids(
            pos_pool,
            self.shadow_train_pos_per_model * max(len(self.shadow_models), 1)
        )
        neg_ids = self._sample_ids(
            neg_pool,
            self.shadow_train_neg_per_model * max(len(self.shadow_models), 1)
        )

        X = []
        y = []
        per_shadow_rows = []

        if len(self.shadow_models) < 2:
            self.attack_head = None
            self.attack_feature_keys = None
            self.attack_head_meta = {
                "attack_head_type": "none_fallback_to_legacy",
                "reason": "need_at_least_2_shadow_models",
                "num_attack_train_rows": 0,
                "num_attack_train_pos": 0,
                "num_attack_train_neg": 0,
                "per_shadow_rows": [],
            }
            return

        for j in range(len(self.shadow_models)):
            target_shadow = self.shadow_models[j]
            ref_models = [m for idx, m in enumerate(self.shadow_models) if idx != j]

            added_pos = 0
            added_neg = 0

            for sid in pos_ids:
                x, lab = self._get_dataset_item_by_id(sid)
                feat = self._extract_feature_dict_for_model(
                    target_model=target_shadow,
                    reference_models=ref_models,
                    x=x,
                    y=lab,
                )
                X.append(feat)
                y.append(1)
                added_pos += 1

            for sid in neg_ids:
                x, lab = self._get_dataset_item_by_id(sid)
                feat = self._extract_feature_dict_for_model(
                    target_model=target_shadow,
                    reference_models=ref_models,
                    x=x,
                    y=lab,
                )
                X.append(feat)
                y.append(0)
                added_neg += 1

            per_shadow_rows.append({
                "shadow_index": int(j),
                "num_pos_used": int(added_pos),
                "num_neg_used": int(added_neg),
            })

        if len(X) < 20 or len(set(y)) < 2:
            self.attack_head = None
            self.attack_feature_keys = None
            self.attack_head_meta = {
                "attack_head_type": "none_fallback_to_legacy",
                "reason": "not_enough_attack_train_rows",
                "num_attack_train_rows": int(len(X)),
                "num_attack_train_pos": int(sum(y)),
                "num_attack_train_neg": int(len(y) - sum(y)),
                "per_shadow_rows": per_shadow_rows,
            }
            return

        feature_keys = sorted([
            k for k, v in X[0].items()
            if isinstance(v, (int, float, np.integer, np.floating))
            and k not in {
                "target_label",
                "pred_clean",
                "legacy_score",
                "stage1_score",
                "binary_pred",
                "iris_score",
            }
        ])

        X_mat = np.array([[float(row.get(k, 0.0)) for k in feature_keys] for row in X], dtype=float)
        y_vec = np.array(y, dtype=int)

        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=self.attack_head_seed,
            n_jobs=-1,
        )
        clf.fit(X_mat, y_vec)

        importances = clf.feature_importances_
        pairs = [(feature_keys[i], float(importances[i])) for i in range(len(feature_keys))]
        pairs = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)

        self.attack_head = clf
        self.attack_feature_keys = feature_keys
        self.attack_head_meta = {
            "attack_head_type": "random_forest_shadow_trained",
            "attack_train_source": "target_split_pools_projected_through_shadow_models",
            "num_attack_train_rows": int(len(X_mat)),
            "num_attack_train_pos": int(np.sum(y_vec == 1)),
            "num_attack_train_neg": int(np.sum(y_vec == 0)),
            "per_shadow_rows": per_shadow_rows,
            "top_feature_importances": pairs[:20],
        }

    def score_sample(self, x: torch.Tensor, y: int) -> Dict[str, Any]:
        feat = self._extract_feature_dict_for_model(
            target_model=self.model,
            reference_models=self.shadow_models,
            x=x,
            y=y,
        )

        legacy_score = float(feat["legacy_score"])

        if self.attack_head is not None and self.attack_feature_keys is not None:
            row = np.array([[float(feat.get(k, 0.0)) for k in self.attack_feature_keys]], dtype=float)
            prob = float(self.attack_head.predict_proba(row)[0, 1])
            iris_score = prob
            score_source = "shadow_trained_random_forest_probability"
        else:
            # fallback
            iris_score = legacy_score
            score_source = "legacy_handcrafted_score"

        pred_binary = int(iris_score >= self.decision_threshold)

        out = {
            **feat,
            "iris_score": float(iris_score),
            "score_source": score_source,
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
        # offline learned head from shadows
        self._fit_attack_head_from_shadows()

        return {
            "unlearn": self.run_group(target_groups["unlearn"], target_idxs["unlearn"], "unlearn"),
            "retain": self.run_group(target_groups["retain"], target_idxs["retain"], "retain"),
            "test": self.run_group(target_groups["test"], target_idxs["test"], "test"),
        }