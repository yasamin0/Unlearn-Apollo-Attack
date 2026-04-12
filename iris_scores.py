from typing import List


def compute_flip_rate(pred_list: List[int], clean_pred: int) -> float:
    """
    Fraction of perturbed predictions that differ from the clean prediction.
    """
    if len(pred_list) == 0:
        return 0.0

    flips = sum(int(p != clean_pred) for p in pred_list)
    return flips / len(pred_list)


def compute_iris_score(
    flip_small: float,
    flip_large: float,
    mode: str = "difference",
) -> float:
    """
    Compute the final IRIS score from small- and large-radius flip rates.

    Supported modes:
    - difference: flip_large - flip_small
    - sum:        flip_large + flip_small
    - large_only: flip_large
    - small_only: flip_small
    """
    if mode == "difference":
        return float(flip_large - flip_small)
    if mode == "sum":
        return float(flip_large + flip_small)
    if mode == "large_only":
        return float(flip_large)
    if mode == "small_only":
        return float(flip_small)

    raise ValueError(f"Unknown iris score mode: {mode}")