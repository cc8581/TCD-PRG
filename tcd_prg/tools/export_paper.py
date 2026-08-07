"""Aggregate evaluation directories into paper tables using standard metrics only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tcd_prg.evaluators.protocols import metric_protocol

CORE_METRICS = (
    "standard_region_miou",
    "standard_task_relation_ng_mean_recall_at_50",
    "standard_verifier_overall_average_precision",
)


def load_run(path: Path) -> tuple[str, dict[str, float]]:
    payload = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    configuration = payload.get("config", {})
    name = configuration.get("name") or path.name
    metrics = payload["summary"]["metrics"]
    return str(name), {
        key: float(value["mean"]) for key, value in metrics.items() if value.get("mean") is not None
    }


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Evaluation directories containing metrics.json")
    parser.add_argument("--output-dir", default="outputs/paper")
    parser.add_argument("--metrics", nargs="*", default=list(CORE_METRICS))
    args = parser.parse_args()

    # No escape hatch: paper tables may contain audited standard metrics only.
    for metric in args.metrics:
        metric_protocol(metric)

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for value in args.runs:
        path = Path(value)
        name, metrics = load_run(path)
        grouped[name].append(metrics)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name, seeds in sorted(grouped.items()):
        row: dict[str, Any] = {"method": name, "seeds": len(seeds)}
        for metric in args.metrics:
            values = np.asarray([seed[metric] for seed in seeds if metric in seed], float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)

    columns = ["method", "seeds"] + [
        item for metric in args.metrics for item in (f"{metric}_mean", f"{metric}_std")
    ]
    with (output / "paper_table.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    latex = [
        "\\begin{tabular}{l" + "c" * len(args.metrics) + "}",
        "\\toprule",
        "Method & " + " & ".join(latex_escape(item) for item in args.metrics) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = []
        for metric in args.metrics:
            mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
            values.append("--" if not np.isfinite(mean) else f"{mean:.3f} $\\pm$ {std:.3f}")
        latex.append(latex_escape(str(row["method"])) + " & " + " & ".join(values) + " \\\\")
    latex.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "paper_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
