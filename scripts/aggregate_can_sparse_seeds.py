#!/usr/bin/env python
"""Aggregate Can sparse K=8 multi-seed rollout results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "p1a_can_sparse_seeds"


SEED_PATTERNS = {
    0: "outputs/eval/can_ph_fm_dct_sparse_k8_*/env/metrics.json",
    1: "outputs/eval/can_ph_fm_dct_sparse_k8_seed1_*/env/metrics.json",
    2: "outputs/eval/can_ph_fm_dct_sparse_k8_seed2_*/env/metrics.json",
}

SUMMARY_KEYS = ("success_rate", "val_action_mse", "smoothness", "delta_action_mse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR.relative_to(PROJECT_ROOT)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    metrics_path = output_dir / "metrics.json"
    figure_path = output_dir / "figures" / "seed_variance.png"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = [_load_seed(seed, pattern) for seed, pattern in SEED_PATTERNS.items()]
    summary = _summarize(seeds)
    payload = {
        "run": {
            "name": "p1a_can_sparse_seeds",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "std": "sample_std_ddof_1",
        },
        "seeds": seeds,
        "summary": summary,
    }
    _write_json(payload, metrics_path)
    _plot_seed_variance(seeds, summary, figure_path)

    print(f"Metrics: {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {figure_path.relative_to(PROJECT_ROOT)}")
    print(
        "Success mean +/- std: "
        f"{summary['success_rate']['mean']:.3f} +/- {summary['success_rate']['std']:.3f}"
    )


def _load_seed(seed: int, pattern: str) -> dict[str, Any]:
    matches = sorted(PROJECT_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    if seed == 0:
        matches = [path for path in matches if "_seed" not in path.parents[1].name]
    if not matches:
        raise FileNotFoundError(f"No metrics found for seed {seed}: {pattern}")
    path = matches[-1]
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    val = data["best_validation"]
    sim = data["simulation"]
    return {
        "seed": seed,
        "run_id": data["run_id"],
        "metrics_path": _rel(path),
        "checkpoint": data.get("checkpoint"),
        "epoch": val.get("epoch"),
        "num_rollouts": sim.get("num_rollouts"),
        "num_successes": sim.get("num_successes"),
        "success_rate": sim.get("success_rate"),
        "val_action_mse": val.get("action_mse"),
        "smoothness": val.get("smoothness"),
        "delta_action_mse": val.get("delta_action_mse"),
    }


def _summarize(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_seeds": len(seeds)}
    for key in SUMMARY_KEYS:
        values = np.asarray([float(seed[key]) for seed in seeds], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def _plot_seed_variance(seeds: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    seed_ids = [seed["seed"] for seed in seeds]
    success = [seed["success_rate"] for seed in seeds]
    mean = summary["success_rate"]["mean"]
    std = summary["success_rate"]["std"]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(seeds))
    ax.bar(x, success, color="#5b8db8", width=0.6, label="seed")
    ax.errorbar(
        len(seeds) + 0.25,
        mean,
        yerr=std,
        fmt="o",
        color="#222222",
        capsize=5,
        label="mean +/- std",
    )
    ax.axhline(mean, color="#222222", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks([*x, len(seeds) + 0.25])
    ax.set_xticklabels([f"seed {seed_id}" for seed_id in seed_ids] + ["mean"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("Can Sparse Full-Spectrum K=8 Seed Variance")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
