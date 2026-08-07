"""Evaluate executed episodes with verified task-domain metrics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tcd_prg.evaluators.protocols import summarize_executed_episodes


def _load(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        return [dict(item) for item in payload["episodes"]]
    raise ValueError("episode input must be JSONL, a JSON list, or {'episodes': [...]} JSON")


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(float(value)) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Executed episode JSON/JSONL files")
    parser.add_argument("--output", default="outputs/episode_metrics.json")
    args = parser.parse_args()

    episodes: list[dict[str, Any]] = []
    for item in args.inputs:
        episodes.extend(_load(Path(item)))
    metrics = summarize_executed_episodes(episodes)
    payload = {
        "protocols": [
            "VPG Completion/Grasp Success/Action Efficiency",
            "executed task-oriented grasp task success rate (when task_grasp_trial is present)",
        ],
        "episode_count": len(episodes),
        "metrics": {key: _json_number(value) for key, value in metrics.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
