import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_iris_binary_summary(summary):
    """
    Binary task:
        positive class = unlearn
        negative class = non-unlearn = retain + test
    """

    scores = []
    y_true = []
    group_names = []

    for sid, item in summary["unlearn"].items():
        scores.append(float(item["iris_score"]))
        y_true.append(1)
        group_names.append("unlearn")

    for sid, item in summary["retain"].items():
        scores.append(float(item["iris_score"]))
        y_true.append(0)
        group_names.append("retain")

    for sid, item in summary["test"].items():
        scores.append(float(item["iris_score"]))
        y_true.append(0)
        group_names.append("test")

    scores = np.array(scores, dtype=float)
    y_true = np.array(y_true, dtype=int)
    group_names = np.array(group_names)

    thresholds = np.percentile(scores, np.linspace(0, 100, 201))

    best_acc = -1.0
    best_th = None
    best_stats = None

    for th in thresholds:
        y_pred = (scores >= th).astype(int)

        acc = float(np.mean(y_pred == y_true))

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))

        tpr_unlearn = float(tp / max(tp + fn, 1))
        tnr_nonunlearn = float(tn / max(tn + fp, 1))
        fpr_nonunlearn = float(fp / max(fp + tn, 1))

        retain_mask = (group_names == "retain")
        test_mask = (group_names == "test")
        non_mask = (y_true == 0)

        retain_as_non = float(np.mean(y_pred[retain_mask] == 0)) if np.sum(retain_mask) > 0 else 0.0
        test_as_non = float(np.mean(y_pred[test_mask] == 0)) if np.sum(test_mask) > 0 else 0.0

        if acc > best_acc:
            best_acc = acc
            best_th = float(th)
            best_stats = {
                "best_acc": float(acc),
                "best_threshold": float(th),
                "tpr_unlearn": float(tpr_unlearn),
                "tnr_nonunlearn": float(tnr_nonunlearn),
                "fpr_nonunlearn": float(fpr_nonunlearn),
                "retain_as_nonunlearn": float(retain_as_non),
                "test_as_nonunlearn": float(test_as_non),
                "tp": tp,
                "fn": fn,
                "tn": tn,
                "fp": fp,
            }

    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.5

    return {
        "auc_unlearn_vs_nonunlearn": auc,
        "best_metrics": best_stats,
        "num_unlearn": int(np.sum(y_true == 1)),
        "num_nonunlearn": int(np.sum(y_true == 0)),
        "scores": scores.tolist(),
        "y_true": y_true.tolist(),
        "group_names": group_names.tolist(),
    }