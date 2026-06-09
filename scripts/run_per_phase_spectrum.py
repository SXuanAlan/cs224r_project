#!/usr/bin/env python
"""Compute transition-conditioned full-spectrum DCT energy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fgac.analysis.frequency_metrics import per_phase_spectrum
from fgac.data.normalization import fit_minmax
from fgac.data.robomimic_hdf5 import build_action_chunks, load_actions, split_demos
from fgac.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/analysis/can_frequency_diagnostic.yaml")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--output-root", default="outputs/analysis/p1b_can_phase_spectrum")
    parser.add_argument("--prefix", default="can_phase_spectrum")
    parser.add_argument("--title-task", default="Can")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(PROJECT_ROOT / args.config)
    chunks, groups, gripper_dim, tau = _load_chunks(cfg, args.tau)
    metrics = per_phase_spectrum(
        chunks,
        gripper_dim_index=gripper_dim,
        tau=tau,
        channel_groups=groups,
    )
    metrics["dataset"] = {
        "path": str((PROJECT_ROOT / cfg["dataset"]["path"]).resolve()),
        "num_chunks": int(chunks.shape[0]),
        "horizon": int(chunks.shape[1]),
        "action_dim": int(chunks.shape[2]),
    }
    metrics["selection_note"] = "Actions are normalized with the same per-dim minmax convention as frequency diagnostics."

    out_root = PROJECT_ROOT / args.output_root
    metrics_dir = out_root / "metrics"
    figures_dir = out_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{args.prefix}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    _plot_aggregate(metrics, figures_dir / f"{args.prefix}_aggregate.png", args.title_task)
    _plot_per_channel(metrics, figures_dir / f"{args.prefix}_per_channel.png", args.title_task)
    print(metrics_path.relative_to(PROJECT_ROOT))
    print((figures_dir / f"{args.prefix}_aggregate.png").relative_to(PROJECT_ROOT))
    print((figures_dir / f"{args.prefix}_per_channel.png").relative_to(PROJECT_ROOT))


def _load_chunks(cfg: dict[str, Any], tau_override: float | None) -> tuple[np.ndarray, dict[str, list[int]], int, float]:
    dataset_path = PROJECT_ROOT / cfg["dataset"]["path"]
    action_data = load_actions(
        dataset_path,
        source=cfg["action"]["source"],
        action_dict_keys=cfg["action"]["action_dict_keys"],
        legacy_key=cfg["action"]["legacy_key"],
        limit_demos=cfg["dataset"].get("limit_demos"),
    )
    train_demos, _ = split_demos(action_data.demos, float(cfg["dataset"]["val_fraction"]), int(cfg["run"]["seed"]))
    norm_stats = fit_minmax(
        [action_data.actions_by_demo[demo] for demo in train_demos],
        eps=float(cfg["normalization"]["eps"]),
    )
    normalized = {demo: norm_stats.normalize(actions) for demo, actions in action_data.actions_by_demo.items()}
    chunk_data = build_action_chunks(
        normalized,
        action_data.demos,
        horizon=int(cfg["chunking"]["horizon"]),
        stride=int(cfg["chunking"]["stride"]),
    )
    groups = {name: [int(v) for v in dims] for name, dims in action_data.groups.items()}
    if cfg.get("transition", {}).get("gripper_dim", "auto") == "auto":
        gripper_dim = groups["gripper"][0]
    else:
        gripper_dim = int(cfg["transition"]["gripper_dim"])
    tau = float(tau_override if tau_override is not None else cfg.get("transition", {}).get("binary_delta_threshold", 1.0))
    return chunk_data.chunks, groups, gripper_dim, tau


def _plot_aggregate(metrics: dict[str, Any], path: Path, title_task: str) -> None:
    x = np.arange(len(metrics["phases"]["transition"]["aggregate"]))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for phase, label in [("transition", "transition"), ("non_transition", "non-transition")]:
        ax.plot(x, metrics["phases"][phase]["aggregate"], marker="o", label=label)
    ax.axvline(7.5, linestyle="--", color="black", linewidth=1.0, label="K=8 cutoff")
    ax.set_title(f"{title_task} DCT Spectrum by Gripper-Transition Phase")
    ax.set_xlabel("DCT frequency index k")
    ax.set_ylabel("Mean normalized energy")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_per_channel(metrics: dict[str, Any], path: Path, title_task: str) -> None:
    groups = ["translation", "rotation", "gripper"]
    x = np.arange(len(metrics["phases"]["transition"]["aggregate"]))
    fig, axes = plt.subplots(len(groups), 1, figsize=(7.0, 8.0), sharex=True)
    for ax, group in zip(axes, groups, strict=True):
        for phase, label in [("transition", "transition"), ("non_transition", "non-transition")]:
            ax.plot(x, metrics["phases"][phase]["per_channel"][group], marker="o", label=label)
        ax.axvline(7.5, linestyle="--", color="black", linewidth=1.0)
        ax.set_title(group)
        ax.set_ylabel("Mean normalized energy")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("DCT frequency index k")
    axes[0].legend()
    fig.suptitle(f"{title_task} Per-Channel DCT Spectrum by Phase", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
