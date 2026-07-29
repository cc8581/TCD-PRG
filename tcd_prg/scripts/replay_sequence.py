"""Validate and export one labelled multi-step transition replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcd_prg.config import load_config
from tcd_prg.runtime import create_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--output", default="outputs/replay_sequence.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=False)
    sequences = adapter.load_sequences(args.scene_id, args.task_index)
    if args.sequence_index >= len(sequences):
        raise IndexError(f"Only {len(sequences)} sequences are available")
    sequence = sequences[args.sequence_index]
    transitions = []
    state_ids = sequence.state_ids.tolist()
    for index, action_id in enumerate(sequence.policy_action_ids.tolist()):
        transitions.append({
            "step": index, "from_state": state_ids[index],
            "action_id": int(action_id), "to_state": state_ids[index + 1],
        })
    payload = {
        "scene_id": args.scene_id, "task_index": args.task_index,
        "sequence_index": args.sequence_index,
        "sequence_topology_valid": sequence.sequence_topology_valid,
        "preparation_action_count": len(sequence.policy_action_ids),
        "within_horizon_h5": len(sequence.policy_action_ids) <= 5,
        "transitions": transitions,
        "terminal_action_ids": sequence.terminal_action_ids.tolist(),
        "final_grasp_source_indices": sequence.final_grasp_source_indices.tolist(),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
