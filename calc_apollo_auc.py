import pickle
import numpy as np
import os

base_runs = [
    "attack_run1",
    "attack_run2",
    "attack_run3",
    "attack_run4",
]

gt_per_class = 50
other_count = 100
classes = ["unlearn", "retain", "test"]

def auc_from_threshold_sweep(d, target_class):
    pts = []
    best_acc = -1
    best_point = None

    for i, fc in enumerate(d["full_classifications"]):
        pred_count = fc["classifications"][target_class]
        tp = fc["correct_classifications"][target_class]
        fp = pred_count - tp

        tpr = tp / gt_per_class
        fpr = fp / other_count
        acc = fc["accuracy"]

        pts.append((fpr, tpr))

        if acc > best_acc:
            best_acc = acc
            best_point = (
                i,
                d["threshold_data"][i],
                fc["accuracy"],
                fc["tpr"][target_class],
            )

    best_by_fpr = {}
    for fpr, tpr in pts:
        if fpr not in best_by_fpr or tpr > best_by_fpr[fpr]:
            best_by_fpr[fpr] = tpr

    xs = np.array(sorted(best_by_fpr.keys()), dtype=float)
    ys = np.array([best_by_fpr[x] for x in xs], dtype=float)

    ys = np.maximum.accumulate(ys)

    if len(xs) == 0 or xs[0] > 0:
        xs = np.insert(xs, 0, 0.0)
        ys = np.insert(ys, 0, 0.0)
    if xs[-1] < 1:
        xs = np.append(xs, 1.0)
        ys = np.append(ys, ys[-1])

    auc = np.trapezoid(ys, xs)
    return auc, xs, ys, best_point

for run_name in base_runs:
    p = rf"save\{run_name}\ResNet18-CIFAR10\perc-0.1-class-None\ternary\Apollo_Offline-GradAscent-Unified-all-results.pkl"

    print("\n" + "#" * 70)
    print("RUN =", run_name)
    print("FILE =", p)

    if not os.path.exists(p):
        print("Status: file not found, skipped.")
        continue

    d = pickle.load(open(p, "rb"))

    for c in classes:
        auc, xs, ys, best_point = auc_from_threshold_sweep(d, c)
        print("-" * 60)
        print("class =", c)
        print("empirical_AUC_one_vs_rest =", round(float(auc), 6))
        print("num_ROC_points =", len(xs))
        print("best_acc_point_idx =", best_point[0])
        print("best_acc_point_threshold =", best_point[1])
        print("best_acc_point_accuracy =", best_point[2])
        print(f"best_acc_point_TPR_{c} =", best_point[3])