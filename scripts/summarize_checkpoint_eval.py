#!/usr/bin/env python
"""Combine checkpoint validation metrics and rollout metrics into one report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to train_flow_matching checkpoint.")
    parser.add_argument("--rollout-json", required=True, help="Path to eval_sim_rollout JSON.")
    parser.add_argument("--output", default=None, help="Output JSON path.")
    parser.add_argument("--status", default="evaluated", help="Run status string to record.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = _resolve(args.checkpoint)
    rollout_path = _resolve(args.rollout_json)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))

    metadata = ckpt.get("metadata", {})
    run = metadata.get("run", {})
    run_id = f"{run.get('name', ckpt_path.parent.name)}_{run.get('timestamp', ckpt_path.parent.name)}"
    output = _resolve(args.output) if args.output else _default_output_path(metadata, run_id)
    output.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "run_id": run_id,
        "status": args.status,
        "checkpoint": _rel(ckpt_path),
        "rollout_json": _rel(rollout_path),
        "task": _infer_task(metadata),
        "dataset": metadata.get("dataset", {}),
        "target": metadata.get("target", {}),
        "model": metadata.get("model", {}),
        "best_validation": {
            "epoch": int(ckpt.get("epoch", -1)),
            **{k: float(v) for k, v in ckpt.get("val_metrics", {}).items()},
        },
        "simulation": rollout.get("summary", {}),
        "simulation_reset_mode": rollout.get("reset_mode"),
        "simulation_horizon": rollout.get("horizon"),
        "simulation_action_exec_horizon": rollout.get("action_exec_horizon"),
        "simulation_residual_lambda": rollout.get("residual_lambda"),
        "videos": [row.get("video_path") for row in rollout.get("rollouts", []) if row.get("video_path")],
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(_rel(output))
    print(json.dumps(report, indent=2))


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _default_output_path(metadata: dict[str, Any], run_id: str) -> Path:
    target_type = metadata.get("target", {}).get("type", "unknown")
    dataset_path = metadata.get("dataset", {}).get("path", "")
    task = _infer_task(metadata)
    return PROJECT_ROOT / "outputs" / "eval" / task / target_type / f"{run_id}_metrics.json"


def _infer_task(metadata: dict[str, Any]) -> str:
    dataset = metadata.get("dataset", {})
    env_id = str(dataset.get("env_id", "")).lower()
    if env_id:
        return env_id.replace("-v1", "").replace("_", "-")
    dataset_path = str(dataset.get("path", "")).lower()
    if "pusht" in dataset_path or "push_t" in dataset_path or "push-t" in dataset_path:
        return "pusht"
    if "/can/" in dataset_path or dataset_path.endswith("can"):
        return "can"
    if "/lift/" in dataset_path or dataset_path.endswith("lift"):
        return "lift"
    if "/square/" in dataset_path or dataset_path.endswith("square"):
        return "square"
    if "tool_hang" in dataset_path or "tool-hang" in dataset_path:
        return "tool-hang"
    return "unknown_task"


if __name__ == "__main__":
    main()
