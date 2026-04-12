import os
import argparse
import time

import numpy as np
from collections import OrderedDict
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt
import ternary

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
import attacks
from models import create_model
from dataset import create_dataset
from attacks.iris_v1 import IRISV1Attack
from attacks.iris_v2 import IRISV2Attack
from attacks.iris_v3 import IRIS_V3
from iris_eval import evaluate_iris_summary
from iris_save import build_iris_paths, save_pickle, save_text, build_iris_report
from iris_sanity import run_basic_iris_sanity_checks

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setminus(A, B):
    return np.array(list(set(A).difference(set(B))))

def get_idx_loaders(split, dataset):
    unlearn_set = dataset.get_subset(split["unlearn"])
    retain_set  = dataset.get_subset(split["retain"])
    test_set    = dataset.get_subset(split["test"])

    unlearn_loader = DataLoader(unlearn_set, batch_size=1, shuffle=False, num_workers=4)
    retain_loader  = DataLoader(retain_set,  batch_size=1, shuffle=False, num_workers=4)
    test_loader    = DataLoader(test_set,    batch_size=1, shuffle=False, num_workers=4)

    idxs    = OrderedDict(unlearn = split["unlearn"], retain = split["retain"], test = split["test"])
    loaders = OrderedDict(unlearn = unlearn_loader,   retain = retain_loader,   test = test_loader)
    return idxs, loaders

def main():
    parser = argparse.ArgumentParser(description='Attack Config')
    parser.add_argument('--data_dir',       type=str,   default='./data',   help='path to dataset')
    parser.add_argument('--dataset',        type=str,   default='',         help='dataset type (default: ImageFolder/ImageTar if empty)')

    parser.add_argument('--model',          type=str,   default='ResNet18', help='model architechture (default: "ResNet18"')
    parser.add_argument('--num_classes',    type=int,   default=None,       help='number of label classes (Model default if None)')
    parser.add_argument('--input_size',     type=int,   default=None,       nargs=3, help='Input all image dimensions (d h w, e.g. --input_size 3 224 224)')
    parser.add_argument('--target_path',    type=str,   default='',         help='Initialize target (unlearned) model from this path (default: none)')

    parser.add_argument('--num_shadow',     type=int,   default=16,         help='number of shadow models (default: 16)')
    parser.add_argument('--shadow_model',   type=str,   default='ResNet18', help='shadow model architechture (default: "ResNet18"')
    parser.add_argument('--shadow_path',    type=str,   default='',         help='Initialize shadow models from this path (default: none)')

    parser.add_argument('--N',              type=int,   default=200,        help='number of samples to attack')
    parser.add_argument('--atk',            type=str,   default='Apollo',   help='Attack Name')
    parser.add_argument('--atk_lr',         type=float, default=1e-1,       help='Attack learning rate')
    parser.add_argument('--atk_epochs',     type=int,   default=30,         help='number of epochs for attack (default: 30)')
    parser.add_argument('--w',              type=float, default=None,       nargs=2, help='Adv. loss function weights')
    parser.add_argument('--eps',            type=float, default=10,         help='epsilon for bound')
    parser.add_argument('--iris_small_radius', type=float, default=0.25)
    parser.add_argument('--iris_large_radius', type=float, default=0.50)
    parser.add_argument('--iris_num_queries_small', type=int, default=20)
    parser.add_argument('--iris_num_queries_large', type=int, default=20)
    parser.add_argument('--iris_score_mode', type=str, default='shadow_sum')
    parser.add_argument('--iris_probe_radius', type=float, default=0.10)
    parser.add_argument('--iris_probe_steps', type=int, default=8)
    parser.add_argument('--iris_probe_samples', type=int, default=12)
    parser.add_argument('--iris_use_relative_score', action='store_true')
    parser.add_argument('--iris_use_early_features', action='store_true')
    parser.add_argument('--iris_early_k', type=int, default=5)
    parser.add_argument('--save_to',        type=str,   required=True,      help='save results to this path')
    parser.add_argument('--debug',                      action="store_true")

    parser.add_argument('--seed',           type=int,   default=42,         help='random seed (default: 42)')
    args = parser.parse_args()


    utils.random_seed(args.seed)

    # Dataloaders
    dataset = create_dataset(dataset_name=args.dataset, setting="Partial", root=args.data_dir, img_size=args.input_size[-1])
    with open(os.path.join(args.target_path, "data_split.pkl"), "rb") as f:
        target_split_orig = pkl.load(f)

    print(target_split_orig["unlearn"][:10], target_split_orig["retain"][:10])
    target_split = {}
    target_split["unlearn"] = np.random.choice(target_split_orig["unlearn"], args.N, replace=False)
    target_split["retain"]  = np.random.choice(target_split_orig["retain"],  args.N, replace=False)
    target_split["test"]    = np.random.choice(target_split_orig["valid"],   args.N, replace=False)
    target_idxs, target_loaders = get_idx_loaders(target_split, dataset)

    # U-MIA Functionality
    if (args.atk == "UMIA"):
        surr_split = {}
        surr_split["unlearn"] = setminus(target_split_orig["unlearn"], target_split["unlearn"])
        surr_split["retain"]  = setminus(target_split_orig["retain"],  target_split["retain"])
        surr_split["test"]    = setminus(target_split_orig["valid"],   target_split["test"])
        surr_split["unlearn"] = np.random.choice(surr_split["unlearn"], args.N, replace=False)
        surr_split["retain"]  = np.random.choice(surr_split["retain"],  args.N, replace=False)
        surr_split["test"]    = np.random.choice(surr_split["test"],    args.N, replace=False)
        surr_idxs, surr_loaders = get_idx_loaders(surr_split, dataset)

    # Target
    target_model = create_model(model_name=args.model, num_classes=args.num_classes)
    target_model.load_state_dict(torch.load(os.path.join(args.target_path, "unlearn.pth.tar"), map_location=DEVICE, weights_only=True))
    target_model.to(DEVICE)
    target_model.eval()

    with open(os.path.join(args.target_path, "unlearn_args.pkl"), "rb") as f:
        unlearn_args = pkl.load(f)
    print("Unlearn Arguments Loaded:", unlearn_args)

    # Shadow models
    shadow_models = nn.ModuleList()
    for i in range(args.num_shadow):
        weights_path = os.path.join(args.shadow_path, f"{i}.pth.tar")
        model = create_model(model_name=args.shadow_model, num_classes=args.num_classes)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
        shadow_models.append(model)

    with open(os.path.join(args.shadow_path, "data_split.pkl"), "rb") as f:
        data_split = pkl.load(f)
    # print(data_split.items())
    print("Models Loaded")

    # IRIS_v3 branch
    if args.atk == "IRIS_v3":
        Atk = IRIS_V3(
            target_model=target_model,
            dataset=dataset,
            shadow_models=shadow_models,
            args=args,
            idxs=target_idxs,
            shadow_col=data_split["shadow_col"],
            unlearn_args=unlearn_args,
        )

        time_col = []
        for name, loader in target_loaders.items():
            print(name)
            for i, (input, label) in enumerate(loader):
                Atk.set_include_exclude(target_idx=target_idxs[name][i])
                start_time = time.time()
                input, label = input.to(DEVICE), label.to(torch.int64).to(DEVICE)
                end_time = time.time()
                time_col.append(end_time - start_time)
                Atk.update_atk_summary(name, input, label, target_idxs[name][i])

        print("Time Used:", np.mean(time_col), np.std(time_col))

        base_path = os.path.join(
            args.save_to,
            f"{unlearn_args.model}-{unlearn_args.dataset}",
            f"perc-{unlearn_args.forget_perc}-class-{unlearn_args.forget_class}"
        )
        summary_path = os.path.join(base_path, "summary")
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)

        with open(os.path.join(summary_path, f"{args.atk}-{unlearn_args.unlearn}.pkl"), "wb") as f:
            pkl.dump(Atk.get_atk_summary(), f)

        ternary_path = os.path.join(base_path, "ternary")
        if not os.path.exists(ternary_path):
            os.makedirs(ternary_path)

        for type_name in Atk.types:
            results = Atk.get_ternary_results(type=type_name)

            if isinstance(results, dict):
                ternary_data = results
            else:
                ternary_points, threshold_data = results
                ternary_data = {"ternary_points": ternary_points, "threshold_data": threshold_data}

            with open(os.path.join(ternary_path, f"{args.atk}-{unlearn_args.unlearn}-{type_name}.pkl"), "wb") as f:
                pkl.dump(ternary_data, f)

            with open(os.path.join(ternary_path, f"{args.atk}-{unlearn_args.unlearn}-{type_name}-all-results.pkl"), "wb") as f:
                pkl.dump(results, f)

        audit_dir = os.path.join(base_path, "query_audit")
        if not os.path.exists(audit_dir):
            os.makedirs(audit_dir)

        audit_json_path = os.path.join(audit_dir, f"{args.atk}-{unlearn_args.unlearn}.json")
        audit_txt_path = os.path.join(audit_dir, f"{args.atk}-{unlearn_args.unlearn}.txt")

        Atk.query_audit.save_json(audit_json_path)
        Atk.query_audit.save_text(audit_txt_path)

        print("Query audit saved to:")
        print(audit_json_path)
        print(audit_txt_path)
        print("Query summary:")
        print(Atk.get_query_audit_summary())

        print("=" * 80)
        print("IRIS_v3 run completed.")
        print("=" * 80)
        return
    
    # IRIS_v1 branch
    if args.atk == "IRIS_v1":
        iris_attack = IRISV1Attack(
            model=target_model,
            shadow_models=shadow_models,
            args=args,
            device=DEVICE,
        )

        target_groups = {
            "unlearn": target_loaders["unlearn"],
            "retain": target_loaders["retain"],
            "test": target_loaders["test"],
        }

        start_time = time.time()
        summary = iris_attack.run(target_groups, target_idxs)
        end_time = time.time()

        print(f"IRIS attack run finished in {end_time - start_time:.4f} seconds")

        eval_results = evaluate_iris_summary(summary)

        forget_class = getattr(unlearn_args, "forget_class", None)
        unlearn_method = getattr(unlearn_args, "unlearn", "GradAscent")

        paths = build_iris_paths(
            save_to=args.save_to,
            model=unlearn_args.model,
            dataset=unlearn_args.dataset,
            forget_perc=unlearn_args.forget_perc,
            class_name=forget_class,
            unlearn_method=unlearn_method,
            attack_name=args.atk,
        )

        save_pickle(summary, paths["summary_path"])
        save_pickle(eval_results, paths["eval_path"])

        run_name = args.save_to.replace("\\", "/").rstrip("/").split("/")[-1]
        report_text = build_iris_report(
            run_name=run_name,
            args=args,
            best_metrics=eval_results["best_metrics"],
        )

        save_text(report_text, paths["report_path"])

        sanity_results = run_basic_iris_sanity_checks(summary=summary, paths=paths)

        report_text += "\n\nSanity checks:\n"
        report_text += f"passed = {sanity_results['passed']}\n"
        report_text += f"group_disjoint = {sanity_results['group_disjoint']['passed']}\n"
        report_text += f"score_lengths = {sanity_results['score_lengths']['passed']}\n"
        if sanity_results["output_files"] is not None:
            report_text += f"output_files = {sanity_results['output_files']['passed']}\n"

        save_text(report_text, paths["report_path"])

        print("=" * 80)
        print("IRIS_v1 run completed.")
        print(f"Summary saved to: {paths['summary_path']}")
        print(f"Eval saved to: {paths['eval_path']}")
        print(f"Report saved to: {paths['report_path']}")
        print("Best metrics:")
        for k, v in eval_results["best_metrics"].items():
            print(f"  {k}: {v}")
        print("Sanity:")
        print(f"  passed: {sanity_results['passed']}")
        print("=" * 80)
        return
    
    # IRIS_v2 branch
    if args.atk == "IRIS_v2":
        iris_attack = IRISV2Attack(
            model=target_model,
            shadow_models=shadow_models,
            args=args,
            device=DEVICE,
        )

        target_groups = {
            "unlearn": target_loaders["unlearn"],
            "retain": target_loaders["retain"],
            "test": target_loaders["test"],
        }

        start_time = time.time()
        summary = iris_attack.run(target_groups, target_idxs)
        end_time = time.time()

        print(f"IRIS_v2 attack run finished in {end_time - start_time:.4f} seconds")

        eval_results = evaluate_iris_summary(summary)

        forget_class = getattr(unlearn_args, "forget_class", None)
        unlearn_method = getattr(unlearn_args, "unlearn", "GradAscent")

        paths = build_iris_paths(
            save_to=args.save_to,
            model=unlearn_args.model,
            dataset=unlearn_args.dataset,
            forget_perc=unlearn_args.forget_perc,
            class_name=forget_class,
            unlearn_method=unlearn_method,
            attack_name=args.atk,
        )

        save_pickle(summary, paths["summary_path"])
        save_pickle(eval_results, paths["eval_path"])

        run_name = args.save_to.replace("\\", "/").rstrip("/").split("/")[-1]
        report_text = build_iris_report(
            run_name=run_name,
            args=args,
            best_metrics=eval_results["best_metrics"],
        )

        save_text(report_text, paths["report_path"])

        sanity_results = run_basic_iris_sanity_checks(summary=summary, paths=paths)

        report_text += "\n\nSanity checks:\n"
        report_text += f"passed = {sanity_results['passed']}\n"
        report_text += f"group_disjoint = {sanity_results['group_disjoint']['passed']}\n"
        report_text += f"score_lengths = {sanity_results['score_lengths']['passed']}\n"
        if sanity_results["output_files"] is not None:
            report_text += f"output_files = {sanity_results['output_files']['passed']}\n"

        save_text(report_text, paths["report_path"])

        print("=" * 80)
        print("IRIS_v2 run completed.")
        print(f"Summary saved to: {paths['summary_path']}")
        print(f"Eval saved to: {paths['eval_path']}")
        print(f"Report saved to: {paths['report_path']}")
        print("Best metrics:")
        for k, v in eval_results["best_metrics"].items():
            print(f"  {k}: {v}")
        print("Sanity:")
        print(f"  passed: {sanity_results['passed']}")
        print("=" * 80)
        return
    
    # Attack!
    Atk = attacks.get_attack(
        name=args.atk,
        target_model=target_model,
        dataset=dataset,
        shadow_models=shadow_models,
        args=args,
        idxs=target_idxs,
        shadow_col=data_split["shadow_col"],
        unlearn_args=unlearn_args,
    )
    if (args.atk == "UMIA"):
        Atk.train_surr(surr_idxs, surr_loaders)
    time_col = []
    for name, loader in target_loaders.items():
        print(name)
        for i, (input, label) in enumerate(pbar := tqdm(loader)):
            Atk.set_include_exclude(target_idx=target_idxs[name][i])
            start_time = time.time()
            input, label = input.to(DEVICE), label.to(torch.int64).to(DEVICE)
            end_time = time.time()
            time_col.append(end_time - start_time)
            Atk.update_atk_summary(name, input, label, target_idxs[name][i])

    print("Time Used:", np.mean(time_col), np.std(time_col))

    # Save Summary
    base_path = os.path.join(
        args.save_to, f"{unlearn_args.model}-{unlearn_args.dataset}",
        f"perc-{unlearn_args.forget_perc}-class-{unlearn_args.forget_class}"
    )
    summary_path = os.path.join(base_path, "summary")
    if (not os.path.exists(summary_path)):
        os.makedirs(summary_path)
    with open(os.path.join(summary_path, f"{args.atk}-{unlearn_args.unlearn}.pkl"), "wb") as f:
        pkl.dump(Atk.get_atk_summary(), f)

    # Interpret results with ternary plots
    ternary_path = os.path.join(base_path, "ternary")
    if (not os.path.exists(ternary_path)):
        os.makedirs(ternary_path)
    for type in Atk.types:
        results = Atk.get_ternary_results(type=type)
        
        if isinstance(results, dict):
            ternary_data = results
        else:
            ternary_points, threshold_data = results
            ternary_data = {"ternary_points": ternary_points, "threshold_data": threshold_data}
        
        with open(os.path.join(ternary_path, f"{args.atk}-{unlearn_args.unlearn}-{type}.pkl"), "wb") as f:
            pkl.dump(ternary_data, f)
        with open(os.path.join(ternary_path, f"{args.atk}-{unlearn_args.unlearn}-{type}-all-results.pkl"), "wb") as f:
            pkl.dump(results, f)

    audit_dir = os.path.join(base_path, "query_audit")
    if not os.path.exists(audit_dir):
        os.makedirs(audit_dir)

    audit_json_path = os.path.join(audit_dir, f"{args.atk}-{unlearn_args.unlearn}.json")
    audit_txt_path = os.path.join(audit_dir, f"{args.atk}-{unlearn_args.unlearn}.txt")

    Atk.query_audit.save_json(audit_json_path)
    Atk.query_audit.save_text(audit_txt_path)

    print("Query audit saved to:")
    print(audit_json_path)
    print(audit_txt_path)
    print("Query summary:")
    print(Atk.get_query_audit_summary())

if __name__ == '__main__':
    main()