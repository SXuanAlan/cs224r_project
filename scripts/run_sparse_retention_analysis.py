#!/usr/bin/env python
"""Analyze which DCT frequency bins sparse top-K retains on Can chunks."""

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

from fgac.data.normalization import fit_minmax
from fgac.data.robomimic_hdf5 import build_action_chunks, load_actions, split_demos
from fgac.transforms.dct import dct_time, topk_frequency_bins
from fgac.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/analysis/can_frequency_diagnostic.yaml")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--output-root", default="outputs/analysis/p1c_can_sparse_retention")
    parser.add_argument("--task-name", default="can_ph")
    parser.add_argument("--title-task", default="Can-PH")
    parser.add_argument("--stdout-prefix", default="[P1.c follow-up]")
    parser.add_argument(
        "--write-count-artifacts",
        action="store_true",
        help="Also rewrite the original count metrics and histogram.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(PROJECT_ROOT / args.config)
    chunks, groups, gripper_dim, tau = _load_chunks(cfg, args.tau)
    z = dct_time(chunks)
    z_sparse, selected = topk_frequency_bins(z, int(args.k))
    transition = np.any(np.abs(np.diff(chunks[..., gripper_dim], axis=1)) > tau, axis=1)
    horizon = int(chunks.shape[1])
    high_freq_cutoff = int(args.k)
    count_metrics = _count_metrics(
        selected=selected,
        transition=transition,
        horizon=horizon,
        k=int(args.k),
        tau=tau,
        gripper_dim=gripper_dim,
        groups=groups,
    )
    magnitude_metrics, magnitude_samples = _magnitude_metrics(
        z_sparse=z_sparse,
        selected=selected,
        transition=transition,
        high_freq_cutoff=high_freq_cutoff,
        k=int(args.k),
        tau=tau,
        gripper_dim=gripper_dim,
        task_name=args.task_name,
    )

    out_root = PROJECT_ROOT / args.output_root
    metrics_dir = out_root / "metrics"
    figures_dir = out_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    count_metrics_path = metrics_dir / "retention.json"
    if args.write_count_artifacts:
        count_metrics_path.write_text(json.dumps(count_metrics, indent=2) + "\n", encoding="utf-8")
        _plot_count_histogram(count_metrics, figures_dir / "sparse_retention_histogram.png", args.title_task)
    elif count_metrics_path.exists():
        with count_metrics_path.open("r", encoding="utf-8") as f:
            count_metrics = json.load(f)

    magnitude_metrics_path = metrics_dir / "retention_magnitude.json"
    magnitude_metrics_path.write_text(json.dumps(magnitude_metrics, indent=2) + "\n", encoding="utf-8")
    magnitude_figure_path = figures_dir / "sparse_retention_magnitude.png"
    combined_figure_path = figures_dir / "sparse_retention_combined.png"
    _plot_magnitude_bars(magnitude_samples, magnitude_figure_path, high_freq_cutoff, args.title_task)
    _plot_combined(count_metrics, magnitude_samples, combined_figure_path, high_freq_cutoff, args.title_task)

    count_fractions = _high_count_fractions(count_metrics, high_freq_cutoff)
    total_ratio = _ratio(
        magnitude_metrics["transition"]["total_high_freq_magnitude"]["mean"],
        magnitude_metrics["non_transition"]["total_high_freq_magnitude"]["mean"],
    )
    mean_ratio = _ratio(
        magnitude_metrics["transition"]["mean_kept_high_freq_magnitude"]["mean"],
        magnitude_metrics["non_transition"]["mean_kept_high_freq_magnitude"]["mean"],
    )
    print(magnitude_metrics_path.relative_to(PROJECT_ROOT))
    print(magnitude_figure_path.relative_to(PROJECT_ROOT))
    print(combined_figure_path.relative_to(PROJECT_ROOT))
    flag = " [FLAG: total-magnitude ratio < 2]" if total_ratio < 2.0 else ""
    if args.stdout_prefix == "[Square retention]":
        print(
            f"{args.stdout_prefix} count k>=8 fraction T/NT = "
            f"{count_fractions['transition']:.3f}/{count_fractions['non_transition']:.3f}; "
            f"total-magnitude ratio = {total_ratio:.3f}; mean-magnitude ratio = {mean_ratio:.3f}{flag}"
        )
    else:
        print(
            f"{args.stdout_prefix} transition / non_transition total-magnitude ratio = "
            f"{total_ratio:.3f}, mean-magnitude ratio = {mean_ratio:.3f}{flag}"
        )


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


def _count_metrics(
    selected: np.ndarray,
    transition: np.ndarray,
    horizon: int,
    k: int,
    tau: float,
    gripper_dim: int,
    groups: dict[str, list[int]],
) -> dict[str, Any]:
    counts = {
        "transition": np.bincount(selected[transition].reshape(-1), minlength=horizon).astype(int).tolist(),
        "non_transition": np.bincount(selected[~transition].reshape(-1), minlength=horizon).astype(int).tolist(),
    }
    totals = {phase: int(sum(values)) for phase, values in counts.items()}
    probabilities = {
        phase: (np.asarray(values, dtype=np.float64) / max(totals[phase], 1)).tolist()
        for phase, values in counts.items()
    }
    return {
        "selection_rule": "top-k temporal DCT frequency bins by summed coefficient energy across action channels",
        "k": int(k),
        "tau": float(tau),
        "gripper_dim_index": int(gripper_dim),
        "num_chunks": int(transition.shape[0]),
        "transition_chunks": int(np.sum(transition)),
        "non_transition_chunks": int(np.sum(~transition)),
        "transition": counts["transition"],
        "non_transition": counts["non_transition"],
        "counts_per_k": counts,
        "probability_per_k": probabilities,
        "high_index_count_k_ge_8": {
            phase: int(sum(values[8:]))
            for phase, values in counts.items()
        },
        "groups": groups,
    }


def _magnitude_metrics(
    z_sparse: np.ndarray,
    selected: np.ndarray,
    transition: np.ndarray,
    high_freq_cutoff: int,
    k: int,
    tau: float,
    gripper_dim: int,
    task_name: str,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    abs_high = np.abs(z_sparse[:, high_freq_cutoff:, :])
    total_high = np.sum(abs_high, axis=(1, 2))
    action_dim = int(z_sparse.shape[-1])
    num_kept_high = np.sum(selected >= high_freq_cutoff, axis=1) * action_dim
    mean_kept_high = total_high / np.maximum(num_kept_high, 1)
    samples = {
        "transition": {
            "total_high_freq_magnitude": total_high[transition],
            "mean_kept_high_freq_magnitude": mean_kept_high[transition],
        },
        "non_transition": {
            "total_high_freq_magnitude": total_high[~transition],
            "mean_kept_high_freq_magnitude": mean_kept_high[~transition],
        },
    }
    metrics = {
        "config": {
            "task": task_name,
            "H": int(z_sparse.shape[1]),
            "K_sparse": int(k),
            "tau_gripper": float(tau),
            "gripper_dim_index": int(gripper_dim),
        },
        "transition": _phase_magnitude_summary(samples["transition"]),
        "non_transition": _phase_magnitude_summary(samples["non_transition"]),
    }
    return metrics, samples


def _phase_magnitude_summary(samples: dict[str, np.ndarray]) -> dict[str, Any]:
    total = samples["total_high_freq_magnitude"]
    mean_kept = samples["mean_kept_high_freq_magnitude"]
    return {
        "n_chunks": int(total.shape[0]),
        "total_high_freq_magnitude": _mean_median(total),
        "mean_kept_high_freq_magnitude": _mean_median(mean_kept),
    }


def _mean_median(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan")}
    return {"mean": float(np.mean(values)), "median": float(np.median(values))}


def _sem(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("inf") if numerator > 0.0 else float("nan")
    return float(numerator / denominator)


def _high_count_fractions(metrics: dict[str, Any], high_freq_cutoff: int) -> dict[str, float]:
    fractions = {}
    for phase in ["transition", "non_transition"]:
        counts = np.asarray(metrics["counts_per_k"][phase], dtype=np.float64)
        fractions[phase] = float(np.sum(counts[high_freq_cutoff:]) / max(np.sum(counts), 1.0))
    return fractions


def _plot_count_histogram(metrics: dict[str, Any], ax_or_path: Any, title_task: str = "Can") -> None:
    owns_figure = not hasattr(ax_or_path, "bar")
    if owns_figure:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        path = Path(ax_or_path)
    else:
        fig = ax_or_path.figure
        ax = ax_or_path
        path = None
    transition = np.asarray(metrics["counts_per_k"]["transition"], dtype=np.float64)
    non_transition = np.asarray(metrics["counts_per_k"]["non_transition"], dtype=np.float64)
    transition = transition / max(np.sum(transition), 1.0)
    non_transition = non_transition / max(np.sum(non_transition), 1.0)
    x = np.arange(len(transition))
    width = 0.38
    ax.bar(x - width / 2, transition, width=width, label="transition")
    ax.bar(x + width / 2, non_transition, width=width, label="non-transition")
    ax.axvline(7.5, linestyle="--", color="black", linewidth=1.0, label="K=8 cutoff")
    ax.set_title(f"{title_task} Sparse DCT Frequency-Bin Retention")
    ax.set_xlabel("DCT frequency index k")
    ax.set_ylabel("Fraction of retained bins")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    if owns_figure:
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def _plot_magnitude_bars(
    samples: dict[str, dict[str, np.ndarray]],
    path: Path,
    high_freq_cutoff: int,
    title_task: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _plot_phase_bars(
        samples,
        "total_high_freq_magnitude",
        axes[0],
        "Total retained high-frequency magnitude",
        f"Total |z| in kept k$\\geq${high_freq_cutoff} coefficients (per chunk)",
    )
    _plot_phase_bars(
        samples,
        "mean_kept_high_freq_magnitude",
        axes[1],
        "Mean magnitude per retained high-frequency coefficient",
        f"Mean |z| per kept k$\\geq${high_freq_cutoff} coefficient",
    )
    fig.suptitle(f"Magnitude-weighted sparse retention on {title_task} (K={high_freq_cutoff})")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_phase_bars(
    samples: dict[str, dict[str, np.ndarray]],
    key: str,
    ax: Any,
    title: str,
    ylabel: str,
) -> None:
    phases = ["transition", "non_transition"]
    means = [float(np.mean(samples[phase][key])) for phase in phases]
    errors = [_sem(samples[phase][key]) for phase in phases]
    ax.bar(phases, means, yerr=errors, capsize=5, color=["#5b8db8", "#c99558"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def _plot_combined(
    count_metrics: dict[str, Any],
    magnitude_samples: dict[str, dict[str, np.ndarray]],
    path: Path,
    high_freq_cutoff: int,
    title_task: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _plot_count_histogram(count_metrics, axes[0], title_task)
    axes[0].set_title("(a) Count-based retained-bin distribution")
    _plot_phase_bars(
        magnitude_samples,
        "total_high_freq_magnitude",
        axes[1],
        "(b) Total high-frequency magnitude",
        f"Total |z| in kept k$\\geq${high_freq_cutoff} coefficients (per chunk)",
    )
    _plot_phase_bars(
        magnitude_samples,
        "mean_kept_high_freq_magnitude",
        axes[2],
        "(c) Mean magnitude per kept high-frequency coefficient",
        f"Mean |z| per kept k$\\geq${high_freq_cutoff} coefficient",
    )
    fig.suptitle(f"Sparse retention on {title_task} (K={high_freq_cutoff})")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
