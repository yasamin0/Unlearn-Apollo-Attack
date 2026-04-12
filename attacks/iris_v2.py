from typing import Dict, Any, List, Tuple
import random

import torch

from iris_scores import compute_flip_rate


class IRISV2Attack:
    """
    Shadow-relative dual-radius label-only attack.

    For each sample:
    - query target model on clean and perturbed neighbors
    - query each shadow model on the same neighbors
    - compute target flip rates at two radii
    - compute mean shadow flip rates at two radii
    - build relative features:
        delta_small = target_small - shadow_small_mean
        delta_large = target_large - shadow_large_mean
    - define final score from the relative features
    """

    def __init__(self, model, shadow_models, args, device=None):
        self.model = model
        self.shadow_models = shadow_models
        self.args = args
        self.device = device if device is not None else torch.device("cpu")

        self.small_radius = float(args.iris_small_radius)
        self.large_radius = float(args.iris_large_radius)
        self.num_q_small = int(args.iris_num_queries_small)
        self.num_q_large = int(args.iris_num_queries_large)
        self.score_mode = str(getattr(args, "iris_score_mode", "shadow_sum"))

        self.clamp_min = 0.0
        self.clamp_max = 1.0

        self.model.eval()
        for m in self.shadow_models:
            m.eval()

        seed = getattr(args, "seed", 0)
        random.seed(seed)
        torch.manual_seed(seed)

    @torch.no_grad()
    def get_hard_label_from_model(self, model: torch.nn.Module, x: torch.Tensor) -> int:
        """
        x expected shape: [C, H, W] or [1, C, H, W]
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
        return int(pred)

    def sample_neighbors(
        self,
        x: torch.Tensor,
        radius: float,
        num_queries: int,
    ) -> List[torch.Tensor]:
        """
        Generate Gaussian perturbations around x and clamp to valid range.
        """
        neighbors = []
        for _ in range(num_queries):
            noise = torch.randn_like(x) * radius
            x_pert = x + noise
            x_pert = torch.clamp(x_pert, min=self.clamp_min, max=self.clamp_max)
            neighbors.append(x_pert)
        return neighbors

    def compute_model_flip_rates(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        small_neighbors: List[torch.Tensor],
        large_neighbors: List[torch.Tensor],
    ) -> Tuple[int, float, float, List[int], List[int]]:
        """
        Returns:
        - clean_pred
        - flip_small
        - flip_large
        - small_preds
        - large_preds
        """
        clean_pred = self.get_hard_label_from_model(model, x)
        small_preds = [self.get_hard_label_from_model(model, z) for z in small_neighbors]
        large_preds = [self.get_hard_label_from_model(model, z) for z in large_neighbors]

        flip_small = compute_flip_rate(small_preds, clean_pred)
        flip_large = compute_flip_rate(large_preds, clean_pred)

        return clean_pred, float(flip_small), float(flip_large), small_preds, large_preds

    def compute_shadow_statistics(
        self,
        x: torch.Tensor,
        small_neighbors: List[torch.Tensor],
        large_neighbors: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Compute per-shadow flip rates and aggregate their means.
        """
        shadow_flip_small = []
        shadow_flip_large = []

        for shadow_model in self.shadow_models:
            _, fs, fl, _, _ = self.compute_model_flip_rates(
                model=shadow_model,
                x=x,
                small_neighbors=small_neighbors,
                large_neighbors=large_neighbors,
            )
            shadow_flip_small.append(float(fs))
            shadow_flip_large.append(float(fl))

        if len(shadow_flip_small) == 0:
            mean_small = 0.0
            mean_large = 0.0
        else:
            mean_small = float(sum(shadow_flip_small) / len(shadow_flip_small))
            mean_large = float(sum(shadow_flip_large) / len(shadow_flip_large))

        return {
            "shadow_flip_small_all": shadow_flip_small,
            "shadow_flip_large_all": shadow_flip_large,
            "shadow_flip_small_mean": mean_small,
            "shadow_flip_large_mean": mean_large,
        }

    def compute_iris_v2_score(
        self,
        delta_small: float,
        delta_large: float,
        flip_small_target: float,
        flip_large_target: float,
        shadow_small_mean: float,
        shadow_large_mean: float,
    ) -> float:
        """
        Score modes:
        - shadow_sum:        delta_small + delta_large
        - shadow_large_only: delta_large
        - shadow_weighted:   0.5 * delta_small + 1.0 * delta_large
        - target_sum:        flip_small_target + flip_large_target
        """
        if self.score_mode == "shadow_sum":
            return float(delta_small + delta_large)
        elif self.score_mode == "shadow_large_only":
            return float(delta_large)
        elif self.score_mode == "shadow_weighted":
            return float(0.5 * delta_small + 1.0 * delta_large)
        elif self.score_mode == "target_sum":
            return float(flip_small_target + flip_large_target)

        raise ValueError(f"Unknown IRIS_v2 score mode: {self.score_mode}")

    def score_sample(
        self,
        x: torch.Tensor,
        y: int,
    ) -> Dict[str, Any]:
        small_neighbors = self.sample_neighbors(
            x=x,
            radius=self.small_radius,
            num_queries=self.num_q_small,
        )
        large_neighbors = self.sample_neighbors(
            x=x,
            radius=self.large_radius,
            num_queries=self.num_q_large,
        )

        target_clean_pred, flip_small_target, flip_large_target, small_preds_target, large_preds_target = \
            self.compute_model_flip_rates(
                model=self.model,
                x=x,
                small_neighbors=small_neighbors,
                large_neighbors=large_neighbors,
            )

        shadow_stats = self.compute_shadow_statistics(
            x=x,
            small_neighbors=small_neighbors,
            large_neighbors=large_neighbors,
        )

        shadow_small_mean = shadow_stats["shadow_flip_small_mean"]
        shadow_large_mean = shadow_stats["shadow_flip_large_mean"]

        delta_small = float(flip_small_target - shadow_small_mean)
        delta_large = float(flip_large_target - shadow_large_mean)

        iris_score = self.compute_iris_v2_score(
            delta_small=delta_small,
            delta_large=delta_large,
            flip_small_target=flip_small_target,
            flip_large_target=flip_large_target,
            shadow_small_mean=shadow_small_mean,
            shadow_large_mean=shadow_large_mean,
        )

        return {
            "target_label": int(y),
            "pred_clean_target": int(target_clean_pred),

            "flip_small_target": float(flip_small_target),
            "flip_large_target": float(flip_large_target),

            "shadow_flip_small_mean": float(shadow_small_mean),
            "shadow_flip_large_mean": float(shadow_large_mean),

            "delta_small": float(delta_small),
            "delta_large": float(delta_large),

            "iris_score": float(iris_score),

            "small_preds_target": [int(p) for p in small_preds_target],
            "large_preds_target": [int(p) for p in large_preds_target],

            "shadow_flip_small_all": [float(v) for v in shadow_stats["shadow_flip_small_all"]],
            "shadow_flip_large_all": [float(v) for v in shadow_stats["shadow_flip_large_all"]],
        }

    def _unpack_batch(
        self,
        batch,
        fallback_start_id: int = 0,
    ) -> List[Tuple[int, torch.Tensor, int]]:
        """
        Supports:
        1) (sample_ids, xs, ys)
        2) (xs, ys)
        3) [(id, x, y), ...]
        """
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
                    sid = int(sample_ids[i])
                    x = xs[i]
                    y = int(ys[i])
                    items.append((sid, x, y))
                return items

            if len(batch) == 2:
                xs, ys = batch
                bs = len(xs)
                for i in range(bs):
                    sid = fallback_start_id + i
                    x = xs[i]
                    y = int(ys[i])
                    items.append((int(sid), x, y))
                return items

        raise ValueError("Unsupported batch format for IRISV2Attack._unpack_batch")

    def run_group(self, group_loader, group_ids) -> Dict[int, Dict[str, Any]]:
        group_results: Dict[int, Dict[str, Any]] = {}

        sample_counter = 0
        for batch in group_loader:
            items = self._unpack_batch(batch, fallback_start_id=sample_counter)
            for _, x, y in items:
                real_sample_id = int(group_ids[sample_counter])
                group_results[real_sample_id] = self.score_sample(x, y)
                sample_counter += 1

        return group_results

    def run(self, target_groups: Dict[str, Any], target_idxs: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
        return {
            "unlearn": self.run_group(target_groups["unlearn"], target_idxs["unlearn"]),
            "retain": self.run_group(target_groups["retain"], target_idxs["retain"]),
            "test": self.run_group(target_groups["test"], target_idxs["test"]),
        }