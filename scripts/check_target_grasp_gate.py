from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED = {
    "target_crop_purity": 0.85,
    "hard_target_mask_purity_fused": 0.85,
    "camera2_transfer_coverage": 0.95,
    "camera2_mask_transfer_accuracy": 0.95,
    "target_proposal_ratio": 0.80,
}


def _metric_value(payload: dict[str, Any], key: str, default: float) -> float:
    metrics = payload.get("metrics", payload)
    value = metrics.get(key, default)
    if isinstance(value, dict):
        value = value.get("mean", default)
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_json")
    args = parser.parse_args()
    raw = Path(args.metrics_json).read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not records:
            raise SystemExit("Metrics file is empty") from None
        payload = records[-1]
    failed = []
    for key, threshold in REQUIRED.items():
        value = _metric_value(payload, key, float("nan"))
        ok = value == value and value >= threshold
        print(f"{key}: {value:.4f} >= {threshold:.2f}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            failed.append(key)
    for key in ("task_grasp_supervised_rows", "ag_width_targets"):
        value = _metric_value(payload, key, 0.0)
        ok = value > 0
        print(f"{key}: {value:.1f} > 0: {'PASS' if ok else 'FAIL'}")
        if not ok:
            failed.append(key)
    if failed:
        raise SystemExit("Gate failed: " + ", ".join(failed))
    print("Target-grasp gate passed.")


if __name__ == "__main__":
    main()
