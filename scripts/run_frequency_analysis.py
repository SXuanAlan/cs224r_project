#!/usr/bin/env python
"""Run Experiment A: DCT frequency decomposition diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fgac.analysis.frequency_metrics import high_energy_ratio, reconstruction_mse, smoothness, summarize_frequency_metrics
from fgac.analysis.plotting import plot_frequency_diagnostic
from fgac.data.normalization import fit_minmax
from fgac.data.robomimic_hdf5 import build_action_chunks, dataset_summary, load_actions, split_demos
from fgac.transforms.dct import dct_time
from fgac.utils.config import load_yaml, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/analysis/frequency_diagnostic.yaml", help="YAML config path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    cfg = load_yaml(config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg["run"]["name"]
    output_cfg = cfg["outputs"]
    run_dir = PROJECT_ROOT / output_cfg["log_dir"] / f"{run_name}_{timestamp}"
    metrics_dir = PROJECT_ROOT / output_cfg["metrics_dir"]
    figures_dir = PROJECT_ROOT / output_cfg["figures_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = PROJECT_ROOT / cfg["dataset"]["path"]
    action_data = load_actions(
        dataset_path,
        source=cfg["action"]["source"],
        action_dict_keys=cfg["action"]["action_dict_keys"],
        legacy_key=cfg["action"]["legacy_key"],
        limit_demos=cfg["dataset"].get("limit_demos"),
    )
    train_demos, val_demos = split_demos(
        action_data.demos,
        val_fraction=float(cfg["dataset"]["val_fraction"]),
        seed=int(cfg["run"]["seed"]),
    )
    split = cfg["dataset"].get("analysis_split", "all")
    if split == "all":
        analysis_demos = action_data.demos
    elif split == "train":
        analysis_demos = train_demos
    elif split == "val":
        analysis_demos = val_demos
    else:
        raise ValueError(f"Unsupported analysis_split: {split}")

    norm_stats = fit_minmax(
        [action_data.actions_by_demo[demo] for demo in train_demos],
        eps=float(cfg["normalization"]["eps"]),
    )
    normalized_actions = {
        demo: norm_stats.normalize(actions) for demo, actions in action_data.actions_by_demo.items()
    }

    horizon = int(cfg["chunking"]["horizon"])
    stride = int(cfg["chunking"]["stride"])
    k_values = [int(k) for k in cfg["chunking"]["k_values"]]
    chunks = build_action_chunks(normalized_actions, analysis_demos, horizon=horizon, stride=stride)
    z = dct_time(chunks.chunks)
    by_k, spectrum = summarize_frequency_metrics(
        actions=chunks.chunks,
        z=z,
        k_values=k_values,
        groups=action_data.groups,
    )
    transition_metrics = _transition_conditioned_metrics(
        cfg=cfg,
        chunks=chunks,
        normalized_actions=normalized_actions,
        action_groups=action_data.groups,
        z=z,
        k_values=k_values,
    )

    metrics: dict[str, Any] = {
        "run": {
            "name": run_name,
            "timestamp": timestamp,
            "config_path": str(config_path.relative_to(PROJECT_ROOT)),
            "git_commit": _git(["rev-parse", "HEAD"]),
        },
        "dataset": {
            **dataset_summary(action_data),
            "train_demos": len(train_demos),
            "val_demos": len(val_demos),
            "analysis_split": split,
            "analysis_demos": len(analysis_demos),
            "num_chunks": int(chunks.chunks.shape[0]),
        },
        "action": {
            "source": action_data.source,
            "dim_names": action_data.dim_names,
            "groups": action_data.groups,
            "note": "auto fell back to legacy actions if action_dict is unavailable",
        },
        "chunking": {
            "horizon": horizon,
            "stride": stride,
            "k_values": k_values,
        },
        "normalization": norm_stats.to_jsonable(),
        "by_k": by_k,
        "spectrum": spectrum,
        "transition_conditioned": transition_metrics,
    }

    metrics_path = metrics_dir / f"{run_name}_{timestamp}.json"
    csv_path = metrics_dir / f"{run_name}_{timestamp}.csv"
    figure_path = figures_dir / f"{run_name}_{timestamp}.png"
    run_metrics_path = run_dir / "metrics.json"

    _write_json(metrics, metrics_path)
    _write_json(metrics, run_metrics_path)
    _write_csv(by_k, action_data.groups, csv_path)
    save_yaml(cfg, run_dir / "config.yaml")
    (run_dir / "git_status.txt").write_text(_git(["status", "--short"]), encoding="utf-8")
    plot_frequency_diagnostic(metrics, figure_path)

    print(f"Run directory: {run_dir.relative_to(PROJECT_ROOT)}")
    print(f"Metrics JSON: {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Metrics CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {figure_path.relative_to(PROJECT_ROOT)}")
    print(_summary_text(metrics))


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_csv(rows: list[dict[str, Any]], groups: dict[str, list[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "k",
        "reconstruction_mse",
        "raw_smoothness",
        "reconstruction_smoothness",
        "smoothness_ratio_to_raw",
        "delta_action_mse",
        "high_energy_ratio_mean",
        "high_energy_ratio_median",
        "high_energy_ratio_std",
    ]
    for group_name in groups:
        fieldnames.append(f"{group_name}_high_energy_ratio_mean")
        fieldnames.append(f"{group_name}_reconstruction_smoothness")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fieldnames if key in row}
            for group_name in groups:
                flat[f"{group_name}_high_energy_ratio_mean"] = row["groups"][group_name][
                    "high_energy_ratio_mean"
                ]
                flat[f"{group_name}_reconstruction_smoothness"] = row["groups"][group_name][
                    "reconstruction_smoothness"
                ]
            writer.writerow(flat)


def _git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"git unavailable: {exc}"


def _summary_text(metrics: dict[str, Any]) -> str:
    lines = [
        "Summary:",
        f"  dataset: {metrics['dataset']['path']}",
        f"  action source: {metrics['action']['source']} ({len(metrics['action']['dim_names'])}D)",
        f"  demos/chunks: {metrics['dataset']['analysis_demos']} demos, {metrics['dataset']['num_chunks']} chunks",
    ]
    trans = metrics.get("transition_conditioned", {})
    if trans:
        lines.append(
            f"  transition chunks: {trans['near_transition_chunks']} near, "
            f"{trans['away_transition_chunks']} away, threshold={trans['threshold']:.4g}"
        )
    for row in metrics["by_k"]:
        lines.append(
            "  "
            f"K={row['k']:>2}: "
            f"MSE={row['reconstruction_mse']:.6g}, "
            f"smooth={row['reconstruction_smoothness']:.6g}, "
            f"E_high={row['high_energy_ratio_mean']:.4f}"
        )
    return "\n".join(lines)


def _transition_conditioned_metrics(
    cfg: dict[str, Any],
    chunks,
    normalized_actions: dict[str, Any],
    action_groups: dict[str, list[int]],
    z,
    k_values: list[int],
) -> dict[str, Any]:
    if "gripper" not in action_groups or len(action_groups["gripper"]) != 1:
        return {"enabled": False, "reason": "requires exactly one gripper action dimension"}
    gripper_dim = action_groups["gripper"][0]
    transition_cfg = cfg.get("transition", {})
    threshold = _resolve_transition_threshold(
        transition_cfg=transition_cfg,
        normalized_actions=normalized_actions,
        gripper_dim=gripper_dim,
    )
    near_mask = _chunk_transition_mask(
        normalized_actions=normalized_actions,
        demo_names=chunks.demo_names,
        starts=chunks.starts,
        horizon=int(cfg["chunking"]["horizon"]),
        gripper_dim=gripper_dim,
        threshold=threshold,
    )
    away_mask = ~near_mask
    result: dict[str, Any] = {
        "enabled": True,
        "gripper_dim": int(gripper_dim),
        "threshold": float(threshold),
        "near_transition_chunks": int(near_mask.sum()),
        "away_transition_chunks": int(away_mask.sum()),
        "by_k": [],
    }
    for k in k_values:
        z_hat = z.copy()
        z_hat[:, k:, :] = 0.0
        from fgac.transforms.dct import idct_time

        recon = idct_time(z_hat)
        ratios = high_energy_ratio(z, k)
        row = {
            "k": int(k),
            "near_high_energy_ratio_mean": _masked_mean(ratios, near_mask),
            "away_high_energy_ratio_mean": _masked_mean(ratios, away_mask),
            "near_reconstruction_mse": _masked_reconstruction_mse(chunks.chunks, recon, near_mask),
            "away_reconstruction_mse": _masked_reconstruction_mse(chunks.chunks, recon, away_mask),
            "near_smoothness": _masked_smoothness(recon, near_mask),
            "away_smoothness": _masked_smoothness(recon, away_mask),
        }
        result["by_k"].append(row)
    return result


def _resolve_transition_threshold(transition_cfg: dict[str, Any], normalized_actions, gripper_dim: int) -> float:
    configured = transition_cfg.get("threshold", "auto")
    if configured != "auto":
        return float(configured)
    deltas = []
    for actions in normalized_actions.values():
        if actions.shape[0] >= 2:
            deltas.append(np.abs(np.diff(actions[:, gripper_dim])))
    all_deltas = np.concatenate(deltas, axis=0) if deltas else np.zeros(1)
    nonzero = all_deltas[all_deltas > 1e-8]
    binary_threshold = float(transition_cfg.get("binary_delta_threshold", 1.0))
    if nonzero.size > 0 and np.percentile(nonzero, 50) > binary_threshold:
        return binary_threshold
    percentile = float(transition_cfg.get("percentile", 95))
    return float(np.percentile(all_deltas, percentile))


def _chunk_transition_mask(normalized_actions, demo_names, starts, horizon: int, gripper_dim: int, threshold: float):
    mask = np.zeros(len(demo_names), dtype=bool)
    transition_by_demo = {}
    for demo, actions in normalized_actions.items():
        transition_by_demo[demo] = np.abs(np.diff(actions[:, gripper_dim])) > threshold
    for i, (demo, start) in enumerate(zip(demo_names, starts)):
        transitions = transition_by_demo[demo]
        begin = int(start)
        end = min(begin + horizon - 1, transitions.shape[0])
        mask[i] = bool(np.any(transitions[begin:end]))
    return mask


def _masked_mean(values, mask) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(values[mask]))


def _masked_reconstruction_mse(reference, reconstructed, mask) -> float:
    if not np.any(mask):
        return float("nan")
    return reconstruction_mse(reference[mask], reconstructed[mask])


def _masked_smoothness(actions, mask) -> float:
    if not np.any(mask):
        return float("nan")
    return smoothness(actions[mask])


if __name__ == "__main__":
    main()
