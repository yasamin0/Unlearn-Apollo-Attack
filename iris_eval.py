from typing import Dict, Any, List, Tuple
import itertools
import numpy as np

from sklearn.metrics import roc_auc_score


CLASS_ORDER = ["unlearn", "retain", "test"]


def flatten_summary_to_scores(summary: Dict[str, Dict[int, Dict[str, Any]]]) -> Tuple[List[float], List[str]]:
    scores = []
    labels = []

    for cls_name in CLASS_ORDER:
        group = summary.get(cls_name, {})
        for _, item in group.items():
            scores.append(float(item["iris_score"]))
            labels.append(cls_name)

    return scores, labels


def compute_one_vs_rest_auc(
    scores: List[float],
    labels: List[str],
    positive_class: str,
) -> float:
    y_true = np.array([1 if lab == positive_class else 0 for lab in labels], dtype=np.int32)
    y_score = np.array(scores, dtype=np.float32)

    unique = np.unique(y_true)
    if len(unique) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_score))


def classify_by_thresholds(
    score: float,
    t1: float,
    t2: float,
    mapping: Tuple[str, str, str],
) -> str:
    """
    mapping example:
    ("retain", "test", "unlearn")
    """
    if score <= t1:
        return mapping[0]
    elif score <= t2:
        return mapping[1]
    else:
        return mapping[2]


def compute_tpr_per_class(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, float]:
    result = {}
    for cls in CLASS_ORDER:
        idx = [i for i, lab in enumerate(y_true) if lab == cls]
        if len(idx) == 0:
            result[cls] = float("nan")
        else:
            correct = sum(int(y_pred[i] == cls) for i in idx)
            result[cls] = correct / len(idx)
    return result


def sweep_ternary_thresholds(
    scores: List[float],
    labels: List[str],
) -> Dict[str, Any]:
    """
    Simple two-threshold sweep over score values.
    Also tries all 6 class-order mappings.
    """
    scores_np = np.array(scores, dtype=np.float32)
    unique_scores = np.unique(scores_np)

    if len(unique_scores) < 3:
        return {
            "all_results": [],
            "best_result": None,
        }

    candidate_thresholds = unique_scores.tolist()
    all_results = []
    best_result = None
    best_acc = -1.0

    all_mappings = list(itertools.permutations(CLASS_ORDER, 3))

    for i in range(len(candidate_thresholds)):
        for j in range(i + 1, len(candidate_thresholds)):
            t1 = float(candidate_thresholds[i])
            t2 = float(candidate_thresholds[j])

            for mapping in all_mappings:
                preds = [classify_by_thresholds(s, t1, t2, mapping) for s in scores]
                acc = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)
                tpr = compute_tpr_per_class(labels, preds)

                row = {
                    "t1": t1,
                    "t2": t2,
                    "mapping": mapping,
                    "acc": float(acc),
                    "tpr_unlearn": float(tpr["unlearn"]),
                    "tpr_retain": float(tpr["retain"]),
                    "tpr_test": float(tpr["test"]),
                }
                all_results.append(row)

                if acc > best_acc:
                    best_acc = acc
                    best_result = row

    return {
        "all_results": all_results,
        "best_result": best_result,
    }


def evaluate_iris_summary(summary: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    scores, labels = flatten_summary_to_scores(summary)

    auc_unlearn = compute_one_vs_rest_auc(scores, labels, "unlearn")
    auc_retain = compute_one_vs_rest_auc(scores, labels, "retain")
    auc_test = compute_one_vs_rest_auc(scores, labels, "test")

    sweep_results = sweep_ternary_thresholds(scores, labels)
    best = sweep_results["best_result"]

    if best is None:
        best_metrics = {
            "best_acc": float("nan"),
            "best_t1": None,
            "best_t2": None,
            "best_mapping": None,
            "best_tpr_unlearn": float("nan"),
            "best_tpr_retain": float("nan"),
            "best_tpr_test": float("nan"),
            "auc_unlearn": auc_unlearn,
            "auc_retain": auc_retain,
            "auc_test": auc_test,
        }
    else:
        best_metrics = {
            "best_acc": best["acc"],
            "best_t1": best["t1"],
            "best_t2": best["t2"],
            "best_mapping": best["mapping"],
            "best_tpr_unlearn": best["tpr_unlearn"],
            "best_tpr_retain": best["tpr_retain"],
            "best_tpr_test": best["tpr_test"],
            "auc_unlearn": auc_unlearn,
            "auc_retain": auc_retain,
            "auc_test": auc_test,
        }

    return {
        "scores": scores,
        "labels": labels,
        "sweep": sweep_results,
        "best_metrics": best_metrics,
    }