import os
import argparse
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
import ternary

un_abbr_to_full = {
    "GradAscent": "Gradient Ascent",
    "Finetune": "Fine-tuning",
    "BadTeacher": "Bad Teacher",
    "SCRUB": "SCRUB",
    "SalUn": "SalUn",
    "SFRon": "SFR-on",
    "Retrain": "Retraining",
}
atk_abbr_to_full = {
    "ULiRA": "U-LiRA",
    "UMIA": "U-MIA",
    "Apollo": "Apollo",
    "Apollo_Offline": "Apollo (Offline)",
}
atk_zorder = {
    "ULiRA": 2,
    "UMIA": 1,
    "Apollo": 4,
    "Apollo_Offline": 3,
}


def main():
    color_map = {
        "Apollo": "#1f77b4",
        "Apollo_Offline": "#17becf",
        "ULiRA": "#ff7f0e",
        "UMIA": "#2ca02c"
    }
    N = 100
    base_path = "./results/ResNet18-CIFAR10/perc-0.1-class-None/"
    for un in ["GradAscent", "Finetune", "BadTeacher", "SCRUB", "SalUn", "SFRon", "Retrain"]: # 
        fig, ax = plt.subplots(figsize=(12, 10))
        scale = 100
        tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)

        for atk in ["ULiRA", "UMIA", "Apollo", "Apollo_Offline"]:
            for type_ in ["", "Unified"]:
                data_path = os.path.join(base_path, "ternary", f"{atk}-{un}-{type_}.pkl")
                if not os.path.exists(data_path):
                    continue
                with open(data_path, "rb") as f:
                    attack_data = pkl.load(f)
                # print(attack_data)
                points = np.array(attack_data['ternary_points']) * 100
                # Reorder from (unlearn, retain, test) to (retain, test, unlearn) for (left, right, bottom) axes
                points = points[:, [1, 2, 0]]
                
                acc = np.array(attack_data['accuracy_results']) + 1e-8
                alpha_vals = 0.1 + 0.8 * (acc ** 2)
                    
                tax.scatter(points, marker='o', c=color_map[atk], s=50, alpha=alpha_vals, zorder=atk_zorder[atk])
        
        optimal_point = (100/3, 100/3, 100/3)
        tax.scatter([optimal_point], marker='^', s=120, c='firebrick', label='Optimal Reference', zorder=10)
        
        # Add reference lines from vertices to optimal point
        # Lines from optimal point to each axis edge at the 100/3 mark
        tax.line((100/3, 0, 200/3), optimal_point, linewidth=1.5, color='k', linestyle='--', alpha=0.8)  # To left axis (Retained)
        tax.line((200/3, 100/3, 0), optimal_point, linewidth=1.5, color='k', linestyle='--', alpha=0.8)  # To right axis (Test)
        tax.line((0, 200/3, 100/3), optimal_point, linewidth=1.5, color='k', linestyle='--', alpha=0.8)  # To bottom axis (Unlearned)

        # tax.get_axes().set_title("title", pad=20)
        tax.left_axis_label("← Retained (%)", offset=0.1, fontsize=16)
        tax.right_axis_label("← Test (%)", offset=0.1, fontsize=16)
        tax.bottom_axis_label("Unlearned (%) →", offset=-0.06, fontsize=16)

        tax.gridlines(multiple=10, color="gray", alpha=0.5, linewidth=0.5)
        tax.boundary(linewidth=2.0)
        tax.ticks(axis='lbr', linewidth=1, multiple=20, fontsize=10, tick_formats="%.0f")
        tax.clear_matplotlib_ticks()
        ax.set_aspect('equal')


        ax.text(0.02, 0.98, un_abbr_to_full[un], fontsize=24, transform=ax.transAxes, verticalalignment='top')
        for atk in ["ULiRA", "UMIA", "Apollo", "Apollo_Offline"]:
            ax.scatter([], [], c=color_map[atk], marker='o', alpha=1.0, s=50, label=atk_abbr_to_full[atk], zorder=atk_zorder[atk])
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.92), fontsize=10)
        # save to no padding on side
        plt.savefig(os.path.join(base_path, "figs", f"{un}.pdf"), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()
