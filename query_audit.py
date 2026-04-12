import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class SampleQueryRecord:
    group_name: str
    sample_id: int
    target_queries: int = 0
    shadow_forwards: int = 0
    optimizer_steps: int = 0
    time_seconds: float = 0.0


class QueryAudit:
    def __init__(self):
        self.records: List[SampleQueryRecord] = []
        self._current: Optional[SampleQueryRecord] = None
        self._tic: Optional[float] = None

    def start_sample(self, group_name: str, sample_id: int):
        if self._current is not None:
            raise RuntimeError("QueryAudit.start_sample called before ending previous sample.")
        self._current = SampleQueryRecord(group_name=group_name, sample_id=int(sample_id))
        self._tic = time.perf_counter()

    def add_target(self, n: int = 1):
        if self._current is not None:
            self._current.target_queries += int(n)

    def add_shadow(self, n: int = 1):
        if self._current is not None:
            self._current.shadow_forwards += int(n)

    def add_steps(self, n: int = 1):
        if self._current is not None:
            self._current.optimizer_steps += int(n)

    def end_sample(self):
        if self._current is None:
            raise RuntimeError("QueryAudit.end_sample called without active sample.")
        toc = time.perf_counter()
        self._current.time_seconds = float(toc - self._tic)
        self.records.append(self._current)
        self._current = None
        self._tic = None

    def summary(self) -> Dict:
        if len(self.records) == 0:
            return {
                "num_samples": 0,
                "mean_target_queries": 0.0,
                "mean_shadow_forwards": 0.0,
                "mean_optimizer_steps": 0.0,
                "mean_time_seconds": 0.0,
                "total_target_queries": 0,
                "total_shadow_forwards": 0,
                "total_optimizer_steps": 0,
                "total_time_seconds": 0.0,
                "by_group": {},
            }

        def _mean(vals):
            return float(sum(vals) / len(vals)) if len(vals) > 0 else 0.0

        by_group = {}
        for g in ["unlearn", "retain", "test"]:
            rows = [r for r in self.records if r.group_name == g]
            if len(rows) == 0:
                continue
            by_group[g] = {
                "num_samples": len(rows),
                "mean_target_queries": _mean([r.target_queries for r in rows]),
                "mean_shadow_forwards": _mean([r.shadow_forwards for r in rows]),
                "mean_optimizer_steps": _mean([r.optimizer_steps for r in rows]),
                "mean_time_seconds": _mean([r.time_seconds for r in rows]),
                "total_target_queries": int(sum(r.target_queries for r in rows)),
                "total_shadow_forwards": int(sum(r.shadow_forwards for r in rows)),
                "total_optimizer_steps": int(sum(r.optimizer_steps for r in rows)),
                "total_time_seconds": float(sum(r.time_seconds for r in rows)),
            }

        out = {
            "num_samples": len(self.records),
            "mean_target_queries": _mean([r.target_queries for r in self.records]),
            "mean_shadow_forwards": _mean([r.shadow_forwards for r in self.records]),
            "mean_optimizer_steps": _mean([r.optimizer_steps for r in self.records]),
            "mean_time_seconds": _mean([r.time_seconds for r in self.records]),
            "total_target_queries": int(sum(r.target_queries for r in self.records)),
            "total_shadow_forwards": int(sum(r.shadow_forwards for r in self.records)),
            "total_optimizer_steps": int(sum(r.optimizer_steps for r in self.records)),
            "total_time_seconds": float(sum(r.time_seconds for r in self.records)),
            "by_group": by_group,
        }
        return out

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "summary": self.summary(),
            "records": [asdict(r) for r in self.records],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def save_text(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s = self.summary()
        lines = []
        lines.append("Query Audit Summary")
        lines.append("")
        lines.append(f"num_samples = {s['num_samples']}")
        lines.append(f"mean_target_queries = {s['mean_target_queries']}")
        lines.append(f"mean_shadow_forwards = {s['mean_shadow_forwards']}")
        lines.append(f"mean_optimizer_steps = {s['mean_optimizer_steps']}")
        lines.append(f"mean_time_seconds = {s['mean_time_seconds']}")
        lines.append(f"total_target_queries = {s['total_target_queries']}")
        lines.append(f"total_shadow_forwards = {s['total_shadow_forwards']}")
        lines.append(f"total_optimizer_steps = {s['total_optimizer_steps']}")
        lines.append(f"total_time_seconds = {s['total_time_seconds']}")
        lines.append("")
        lines.append("By group:")
        for g, row in s["by_group"].items():
            lines.append(f"[{g}]")
            for k, v in row.items():
                lines.append(f"{k} = {v}")
            lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))