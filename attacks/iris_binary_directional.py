import math
from typing import Dict, Any, List

import numpy as np
import torch

from query_audit import QueryAudit

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IRISBinaryDirectional:
    """
    Thesis-aligned IRIS:
    - binary
    - label-only
    - query-efficient
    - directional radius probing

    Binary task:
        unlearn  vs  non-unlearn
    where non-unlearn = retain ∪ test
    """

    def __init__(self, model, args, device=None):
        self.model = model
        self.args = args
        self.device = device if device is not None else DEVICE

        self.num_directions = int(getattr(args, "iris_num_directions", 8))
        self.radius_min = float(getattr(args, "iris_radius_min", 0.02))
        self.radius_max = float(getattr(args, "iris_radius_max", 0.30))
        self.radius_steps = int(getattr(args, "iris_radius_steps", 8))
        self.decision_threshold = float(getattr(args, "iris_binary_threshold", 0.50))

        self.clamp_min = 0.0
        self.clamp_max = 1.0

        self.model.eval()

        self.query_audit = QueryAudit()

    # ------------------------------------------------------------
    # Query-audit helpers
    # ------------------------------------------------------------
    def _qa_start(self, group_name, sample_id):
        self.query_audit.start_sample(group_name, int(sample_id))

    def _qa_target(self, n=1):
        self.query_audit.add_target(n)

    def _qa_end(self):
        self.query_audit.end_sample()

    def get_query_audit_summary(self):
        return self.query_audit.summary()

    # ------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------
    @torch.no_grad()
    def get_hard_label(self, x: torch.Tensor) -> int:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        self._qa_target(1)
        logits = self.model(x)
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

    def probe_direction(
        self,
        x: torch.Tensor,
        clean_label: int,
        direction: torch.Tensor,
        radius_schedule: List[float],
    ) -> Dict[str, Any]:
        """
        For one direction:
        - find first flip radius
        - measure persistence after first flip
        - mark no-flip if no change occurred
        """
        labels_along_path = []
        first_flip_radius = None
        first_flip_idx = None

        for i, r in enumerate(radius_schedule):
            z = x + r * direction
            z = torch.clamp(z, min=self.clamp_min, max=self.clamp_max)
            pred = self.get_hard_label(z)
            labels_along_path.append(int(pred))

            if first_flip_radius is None and pred != clean_label:
                first_flip_radius = float(r)
                first_flip_idx = i

        no_flip = first_flip_radius is None

        if no_flip:
            persistence = 0.0
            alt_label = None
        else:
            suffix = labels_along_path[first_flip_idx:]
            changed = [int(v != clean_label) for v in suffix]
            persistence = float(sum(changed) / len(changed)) if len(changed) > 0 else 0.0

            alt_candidates = [v for v in suffix if v != clean_label]
            if len(alt_candidates) == 0:
                alt_label = None
            else:
                vals, counts = np.unique(np.array(alt_candidates), return_counts=True)
                alt_label = int(vals[np.argmax(counts)])

        return {
            "labels_along_path": labels_along_path,
            "first_flip_radius": None if no_flip else float(first_flip_radius),
            "first_flip_idx": None if no_flip else int(first_flip_idx),
            "persistence_after_flip": float(persistence),
            "no_flip": bool(no_flip),
            "dominant_alt_label": alt_label,
        }

    def aggregate_directional_features(
        self,
        direction_results: List[Dict[str, Any]],
        radius_schedule: List[float],
    ) -> Dict[str, float]:
        """
        Aggregate across directions:
        - flip radius summary
        - persistence summary
        - flip spread ratio
        """
        num_dirs = len(direction_results)
        num_no_flip = sum(int(r["no_flip"]) for r in direction_results)
        num_flip = num_dirs - num_no_flip

        flip_radii = [r["first_flip_radius"] for r in direction_results if not r["no_flip"]]
        persistences = [r["persistence_after_flip"] for r in direction_results if not r["no_flip"]]

        max_radius = float(radius_schedule[-1])

        # Smaller first-flip radius => more locally fragile
        mean_first_flip_radius = float(np.mean(flip_radii)) if len(flip_radii) > 0 else max_radius
        min_first_flip_radius = float(np.min(flip_radii)) if len(flip_radii) > 0 else max_radius
        std_first_flip_radius = float(np.std(flip_radii)) if len(flip_radii) > 1 else 0.0

        mean_persistence = float(np.mean(persistences)) if len(persistences) > 0 else 0.0
        max_persistence = float(np.max(persistences)) if len(persistences) > 0 else 0.0

        flip_fraction = float(num_flip / max(num_dirs, 1))
        no_flip_fraction = float(num_no_flip / max(num_dirs, 1))

        # normalized "early flip" score: higher = earlier flip
        early_flip_score = float(1.0 - (mean_first_flip_radius / max_radius)) if max_radius > 0 else 0.0

        # spread ratio: more variety in flip radii => more unstable
        spread_ratio = float(std_first_flip_radius / max_radius) if max_radius > 0 else 0.0

        return {
            "flip_fraction": flip_fraction,
            "no_flip_fraction": no_flip_fraction,
            "mean_first_flip_radius": mean_first_flip_radius,
            "min_first_flip_radius": min_first_flip_radius,
            "std_first_flip_radius": std_first_flip_radius,
            "mean_persistence": mean_persistence,
            "max_persistence": max_persistence,
            "early_flip_score": early_flip_score,
            "flip_spread_ratio": spread_ratio,
        }

    def build_influence_signature(self, agg: Dict[str, float]) -> Dict[str, float]:
        """
        Influence signature phi(x).
        Keep it simple and interpretable.
        """
        phi = {
            "phi_1_flip_fraction": float(agg["flip_fraction"]),
            "phi_2_early_flip": float(agg["early_flip_score"]),
            "phi_3_persistence": float(agg["mean_persistence"]),
            "phi_4_spread": float(agg["flip_spread_ratio"]),
        }
        return phi

    def compute_binary_score(self, phi: Dict[str, float]) -> float:
        """
        Unlearn score.
        Interpretable weighted combination.
        """
        score = (
            0.35 * phi["phi_1_flip_fraction"] +
            0.30 * phi["phi_2_early_flip"] +
            0.25 * phi["phi_3_persistence"] +
            0.10 * phi["phi_4_spread"]
        )
        return float(score)

    def classify_binary(self, score: float) -> int:
        """
        1 = unlearn
        0 = non-unlearn
        """
        return int(score >= self.decision_threshold)

    def score_sample(
        self,
        x: torch.Tensor,
        y: int,
    ) -> Dict[str, Any]:
        clean_label = self.get_hard_label(x)
        radius_schedule = self.build_radius_schedule()

        direction_results = []
        for _ in range(self.num_directions):
            d = self.random_unit_direction(x)
            res = self.probe_direction(
                x=x,
                clean_label=clean_label,
                direction=d,
                radius_schedule=radius_schedule,
            )
            direction_results.append(res)

        agg = self.aggregate_directional_features(direction_results, radius_schedule)
        phi = self.build_influence_signature(agg)
        iris_score = self.compute_binary_score(phi)
        pred_binary = self.classify_binary(iris_score)

        return {
            "target_label": int(y),
            "pred_clean": int(clean_label),

            "radius_schedule": [float(r) for r in radius_schedule],
            "direction_results": direction_results,

            "flip_fraction": float(agg["flip_fraction"]),
            "no_flip_fraction": float(agg["no_flip_fraction"]),
            "mean_first_flip_radius": float(agg["mean_first_flip_radius"]),
            "min_first_flip_radius": float(agg["min_first_flip_radius"]),
            "std_first_flip_radius": float(agg["std_first_flip_radius"]),
            "mean_persistence": float(agg["mean_persistence"]),
            "max_persistence": float(agg["max_persistence"]),
            "early_flip_score": float(agg["early_flip_score"]),
            "flip_spread_ratio": float(agg["flip_spread_ratio"]),

            "phi": phi,
            "iris_score": float(iris_score),
            "binary_pred": int(pred_binary),
        }

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