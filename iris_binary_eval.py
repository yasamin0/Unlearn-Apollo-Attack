import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return 0.5


def _safe_mean(values):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values):
    if len(values) <= 1:
        return 0.0
    return float(np.std(values))


def _collect_feature_keys(summary):
    keys = set()
    for group in ["unlearn", "retain", "test"]:
        for _, item in summary[group].items():
            for k, v in item.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    keys.add(k)
    banned = {"target_label", "pred_clean", "binary_pred"}
    return sorted([k for k in keys if k not in banned])


def _group_feature_stats(summary, feature_keys):
    out = {}
    for group in ["unlearn", "retain", "test"]:
        out[group] = {}
        for key in feature_keys:
            vals = []
            for _, item in summary[group].items():
                if key in item and isinstance(item[key], (int, float, np.integer, np.floating)):
                    vals.append(float(item[key]))
            out[group][key] = {
                "mean": _safe_mean(vals),
                "std": _safe_std(vals),
                "min": float(np.min(vals)) if len(vals) > 0 else 0.0,
                "max": float(np.max(vals)) if len(vals) > 0 else 0.0,
            }
    return out


def _evaluate_one_orientation(scores, y_true, group_names, orientation_name):
    thresholds = np.unique(
        np.concatenate(
            [
                scores,
                np.percentile(scores, np.linspace(0, 100, 401)),
                np.array([scores.min() - 1e-12, scores.max() + 1e-12], dtype=float),
            ]
        )
    )
    thresholds.sort()

    best_bal_acc = -1.0
    best_stats = None

    for th in thresholds:
        y_pred = (scores >= th).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))

        tpr_unlearn = float(tp / max(tp + fn, 1))
        tnr_nonunlearn = float(tn / max(tn + fp, 1))
        fpr_nonunlearn = float(fp / max(fp + tn, 1))
        bal_acc = 0.5 * (tpr_unlearn + tnr_nonunlearn)
        acc = float(np.mean(y_pred == y_true))
        youden_j = float(tpr_unlearn - fpr_nonunlearn)

        retain_mask = (group_names == "retain")
        test_mask = (group_names == "test")

        retain_as_non = float(np.mean(y_pred[retain_mask] == 0)) if np.sum(retain_mask) > 0 else 0.0
        test_as_non = float(np.mean(y_pred[test_mask] == 0)) if np.sum(test_mask) > 0 else 0.0

        candidate = {
            "score_orientation": orientation_name,
            "best_balanced_acc": float(bal_acc),
            "best_acc": float(acc),
            "best_threshold": float(th),
            "tpr_unlearn": float(tpr_unlearn),
            "tnr_nonunlearn": float(tnr_nonunlearn),
            "fpr_nonunlearn": float(fpr_nonunlearn),
            "youden_j": float(youden_j),
            "retain_as_nonunlearn": float(retain_as_non),
            "test_as_nonunlearn": float(test_as_non),
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
        }

        if best_stats is None:
            best_stats = candidate
            best_bal_acc = bal_acc
        else:
            old = best_stats
            better = False
            if bal_acc > best_bal_acc + 1e-12:
                better = True
            elif abs(bal_acc - best_bal_acc) <= 1e-12 and acc > old["best_acc"] + 1e-12:
                better = True
            elif abs(bal_acc - best_bal_acc) <= 1e-12 and abs(acc - old["best_acc"]) <= 1e-12 and abs(youden_j) > abs(old["youden_j"]) + 1e-12:
                better = True

            if better:
                best_stats = candidate
                best_bal_acc = bal_acc

    auc = _safe_auc(y_true, scores)
    return {
        "auc": float(auc),
        "best_metrics": best_stats,
        "scores_used": scores.tolist(),
    }


def evaluate_iris_binary_summary(summary):
    """
    Binary task:
        positive class = unlearn
        negative class = retain + test

    Important improvements:
    - checks BOTH score polarities: score and -score
    - chooses the better orientation
    - returns group-wise feature diagnostics
    """

    raw_scores = []
    y_true = []
    group_names = []
    sample_ids = []

    for sid, item in summary["unlearn"].items():
        raw_scores.append(float(item["iris_score"]))
        y_true.append(1)
        group_names.append("unlearn")
        sample_ids.append(int(sid))

    for sid, item in summary["retain"].items():
        raw_scores.append(float(item["iris_score"]))
        y_true.append(0)
        group_names.append("retain")
        sample_ids.append(int(sid))

    for sid, item in summary["test"].items():
        raw_scores.append(float(item["iris_score"]))
        y_true.append(0)
        group_names.append("test")
        sample_ids.append(int(sid))

    raw_scores = np.array(raw_scores, dtype=float)
    y_true = np.array(y_true, dtype=int)
    group_names = np.array(group_names)
    sample_ids = np.array(sample_ids, dtype=int)

    eval_raw = _evaluate_one_orientation(
        scores=raw_scores,
        y_true=y_true,
        group_names=group_names,
        orientation_name="raw_score_means_more_unlearn",
    )

    eval_neg = _evaluate_one_orientation(
        scores=-raw_scores,
        y_true=y_true,
        group_names=group_names,
        orientation_name="negated_score_means_more_unlearn",
    )

    if eval_raw["best_metrics"]["best_balanced_acc"] > eval_neg["best_metrics"]["best_balanced_acc"] + 1e-12:
        selected = eval_raw
        selected_name = "raw_score_means_more_unlearn"
        selected_scores = raw_scores
    elif eval_neg["best_metrics"]["best_balanced_acc"] > eval_raw["best_metrics"]["best_balanced_acc"] + 1e-12:
        selected = eval_neg
        selected_name = "negated_score_means_more_unlearn"
        selected_scores = -raw_scores
    else:
        if eval_raw["auc"] >= eval_neg["auc"]:
            selected = eval_raw
            selected_name = "raw_score_means_more_unlearn"
            selected_scores = raw_scores
        else:
            selected = eval_neg
            selected_name = "negated_score_means_more_unlearn"
            selected_scores = -raw_scores

    feature_keys = _collect_feature_keys(summary)
    feature_stats = _group_feature_stats(summary, feature_keys)

    score_stats = {
        "raw_score": {
            "unlearn_mean": _safe_mean([float(item["iris_score"]) for _, item in summary["unlearn"].items()]),
            "retain_mean": _safe_mean([float(item["iris_score"]) for _, item in summary["retain"].items()]),
            "test_mean": _safe_mean([float(item["iris_score"]) for _, item in summary["test"].items()]),
            "all_mean": float(np.mean(raw_scores)) if len(raw_scores) > 0 else 0.0,
            "all_std": float(np.std(raw_scores)) if len(raw_scores) > 0 else 0.0,
            "min": float(np.min(raw_scores)) if len(raw_scores) > 0 else 0.0,
            "max": float(np.max(raw_scores)) if len(raw_scores) > 0 else 0.0,
        },
        "selected_orientation_score": {
            "orientation": selected_name,
            "auc": float(selected["auc"]),
            "all_mean": float(np.mean(selected_scores)) if len(selected_scores) > 0 else 0.0,
            "all_std": float(np.std(selected_scores)) if len(selected_scores) > 0 else 0.0,
            "min": float(np.min(selected_scores)) if len(selected_scores) > 0 else 0.0,
            "max": float(np.max(selected_scores)) if len(selected_scores) > 0 else 0.0,
        },
    }

    return {
        "auc_unlearn_vs_nonunlearn": float(selected["auc"]),
        "auc_raw_score": float(eval_raw["auc"]),
        "auc_negated_score": float(eval_neg["auc"]),
        "selected_score_orientation": selected_name,
        "best_metrics": selected["best_metrics"],
        "num_unlearn": int(np.sum(y_true == 1)),
        "num_nonunlearn": int(np.sum(y_true == 0)),
        "raw_scores": raw_scores.tolist(),
        "selected_scores": selected_scores.tolist(),
        "y_true": y_true.tolist(),
        "group_names": group_names.tolist(),
        "sample_ids": sample_ids.tolist(),
        "score_stats": score_stats,
        "feature_keys": feature_keys,
        "feature_stats_by_group": feature_stats,
    }