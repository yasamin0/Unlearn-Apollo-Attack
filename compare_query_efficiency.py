import json
import sys


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["summary"]


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("python compare_query_efficiency.py <apollo_json> <iris_json>")
        return

    apollo = load_summary(sys.argv[1])
    iris = load_summary(sys.argv[2])

    print("=== Query Efficiency Comparison ===")
    print("")
    print("Apollo")
    print(f"mean_target_queries    = {apollo['mean_target_queries']}")
    print(f"mean_shadow_forwards   = {apollo['mean_shadow_forwards']}")
    print(f"mean_optimizer_steps   = {apollo['mean_optimizer_steps']}")
    print(f"mean_time_seconds      = {apollo['mean_time_seconds']}")
    print("")
    print("IRIS")
    print(f"mean_target_queries    = {iris['mean_target_queries']}")
    print(f"mean_shadow_forwards   = {iris['mean_shadow_forwards']}")
    print(f"mean_optimizer_steps   = {iris['mean_optimizer_steps']}")
    print(f"mean_time_seconds      = {iris['mean_time_seconds']}")
    print("")
    print("Ratios (IRIS / Apollo)")
    print(f"target_query_ratio     = {iris['mean_target_queries'] / max(apollo['mean_target_queries'], 1e-12)}")
    print(f"shadow_forward_ratio   = {iris['mean_shadow_forwards'] / max(apollo['mean_shadow_forwards'], 1e-12)}")
    print(f"step_ratio             = {iris['mean_optimizer_steps'] / max(apollo['mean_optimizer_steps'], 1e-12)}")
    print(f"time_ratio             = {iris['mean_time_seconds'] / max(apollo['mean_time_seconds'], 1e-12)}")


if __name__ == "__main__":
    main()