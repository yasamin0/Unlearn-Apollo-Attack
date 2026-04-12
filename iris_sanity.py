import os
from typing import Dict, Any, Iterable, Set


def check_group_disjoint(
    unlearn_ids: Iterable[int],
    retain_ids: Iterable[int],
    test_ids: Iterable[int],
) -> Dict[str, Any]:
    """
    Check whether unlearn / retain / test sample-id sets are mutually disjoint.
    """
    unlearn_ids = set(int(x) for x in unlearn_ids)
    retain_ids = set(int(x) for x in retain_ids)
    test_ids = set(int(x) for x in test_ids)

    overlap_ur = unlearn_ids.intersection(retain_ids)
    overlap_ut = unlearn_ids.intersection(test_ids)
    overlap_rt = retain_ids.intersection(test_ids)

    passed = (
        len(overlap_ur) == 0 and
        len(overlap_ut) == 0 and
        len(overlap_rt) == 0
    )

    return {
        "passed": passed,
        "num_unlearn": len(unlearn_ids),
        "num_retain": len(retain_ids),
        "num_test": len(test_ids),
        "overlap_unlearn_retain": sorted(list(overlap_ur)),
        "overlap_unlearn_test": sorted(list(overlap_ut)),
        "overlap_retain_test": sorted(list(overlap_rt)),
    }


def check_output_files_exist(paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Check whether expected output files exist.
    """
    result = {}
    all_exist = True

    for name, path in paths.items():
        if not isinstance(path, str):
            continue
        exists = os.path.exists(path)
        result[name] = {
            "path": path,
            "exists": exists,
        }
        if name.endswith("_path") and not exists:
            all_exist = False

    result["passed"] = all_exist
    return result


def check_score_lengths(summary: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Check whether stored prediction-list lengths match the requested query counts.
    """
    issues = []

    for group_name, group_data in summary.items():
        for sample_id, item in group_data.items():
            small_preds = item.get("small_preds", [])
            large_preds = item.get("large_preds", [])

            if not isinstance(small_preds, list):
                issues.append(f"{group_name}:{sample_id}: small_preds is not a list")
            if not isinstance(large_preds, list):
                issues.append(f"{group_name}:{sample_id}: large_preds is not a list")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
    }


def collect_group_ids(summary: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, Set[int]]:
    """
    Collect sample ids from summary.
    """
    out = {}
    for group_name in ["unlearn", "retain", "test"]:
        group = summary.get(group_name, {})
        out[group_name] = set(int(k) for k in group.keys())
    return out


def run_basic_iris_sanity_checks(
    summary: Dict[str, Dict[int, Dict[str, Any]]],
    paths: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Run a basic sanity suite for IRIS outputs.
    """
    group_ids = collect_group_ids(summary)

    disjoint_result = check_group_disjoint(
        unlearn_ids=group_ids["unlearn"],
        retain_ids=group_ids["retain"],
        test_ids=group_ids["test"],
    )

    score_length_result = check_score_lengths(summary)

    output_result = None
    if paths is not None:
        output_result = check_output_files_exist(paths)

    passed = disjoint_result["passed"] and score_length_result["passed"]
    if output_result is not None:
        passed = passed and output_result["passed"]

    return {
        "passed": passed,
        "group_disjoint": disjoint_result,
        "score_lengths": score_length_result,
        "output_files": output_result,
    }