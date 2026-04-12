from typing import Dict, Any, List, Tuple
import random

import torch

from iris_scores import compute_flip_rate, compute_iris_score

class IRISV1Attack:
    """
    Simple label-only radius-based attack.

    For each sample:
    - query the clean sample
    - query perturbed neighbors at a small radius
    - query perturbed neighbors at a large radius
    - compute flip rates
    - define iris_score from the two flip rates
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
        self.score_mode = str(args.iris_score_mode)

        self.clamp_min = 0.0
        self.clamp_max = 1.0

        self.model.eval()

        seed = getattr(args, "seed", 0)
        random.seed(seed)
        torch.manual_seed(seed)

    @torch.no_grad()
    def get_hard_label(self, x: torch.Tensor) -> int:
        """
        x expected shape: [C, H, W] or [1, C, H, W]
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        logits = self.model(x)
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

    def score_sample(
        self,
        x: torch.Tensor,
        y: int,
    ) -> Dict[str, Any]:
        clean_pred = self.get_hard_label(x)

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

        small_preds = [self.get_hard_label(z) for z in small_neighbors]
        large_preds = [self.get_hard_label(z) for z in large_neighbors]

        flip_small = compute_flip_rate(small_preds, clean_pred)
        flip_large = compute_flip_rate(large_preds, clean_pred)
        iris_score = compute_iris_score(
            flip_small=flip_small,
            flip_large=flip_large,
            mode=self.score_mode,
        )

        return {
            "target_label": int(y),
            "pred_clean": int(clean_pred),
            "flip_small": float(flip_small),
            "flip_large": float(flip_large),
            "iris_score": float(iris_score),
            "small_preds": [int(p) for p in small_preds],
            "large_preds": [int(p) for p in large_preds],
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

        raise ValueError("Unsupported batch format for IRISV1Attack._unpack_batch")

    def run_group(self, group_loader) -> Dict[int, Dict[str, Any]]:
        group_results: Dict[int, Dict[str, Any]] = {}
        running_id = 0

        for batch in group_loader:
            items = self._unpack_batch(batch, fallback_start_id=running_id)
            for sample_id, x, y in items:
                group_results[int(sample_id)] = self.score_sample(x, y)
                running_id += 1

        return group_results

    def run(self, target_groups: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
        return {
            "unlearn": self.run_group(target_groups["unlearn"]),
            "retain": self.run_group(target_groups["retain"]),
            "test": self.run_group(target_groups["test"]),
        }