#!/usr/bin/env python
"""Run DCT frequency diagnostics for PushT zarr actions."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fgac.analysis.frequency_metrics import summarize_frequency_metrics
from fgac.analysis.plotting import plot_frequency_diagnostic
from fgac.data.normalization import fit_minmax
from fgac.data.pusht_zarr import load_pusht_replay, pusht_dataset_summary
from fgac.data.robomimic_hdf5 import build_action_chunks, split_demos
from fgac.transforms.dct import dct_time
from fgac.utils.config import load_yaml, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/analysis/pusht_frequency_diagnostic.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    cfg = load_yaml(cfg_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg["run"]["name"]
    run_dir = PROJECT_ROOT / cfg["outputs"]["log_dir"] / f"{run_name}_{timestamp}"
    metrics_dir = PROJECT_ROOT / cfg["outputs"]["metrics_dir"]
    figures_dir = PROJECT_ROOT / cfg["outputs"]["figures_dir"]
    for path in [run_dir, metrics_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)

    data = load_pusht_replay(
        PROJECT_ROOT / cfg["dataset"]["path"],
        obs_key=cfg["dataset"].get("obs_key", "state"),
        action_key=cfg["dataset"].get("action_key", "action"),
        limit_demos=cfg["dataset"].get("limit_demos"),
    )
    train_demos, val_demos = split_demos(data.demos, float(cfg["dataset"]["val_fraction"]), int(cfg["run"]["seed"]))
    split = cfg["dataset"].get("analysis_split", "all")
    if split == "all":
        analysis_demos = data.demos
    elif split == "train":
        analysis_demos = train_demos
    elif split == "val":
        analysis_demos = val_demos
    else:
        raise ValueError(f"Unsupported analysis_split: {split}")

    norm = fit_minmax([data.actions_by_demo[d] for d in train_demos], eps=float(cfg["normalization"]["eps"]))
    norm_actions = {demo: norm.normalize(actions) for demo, actions in data.actions_by_demo.items()}
    chunks = build_action_chunks(norm_actions, analysis_demos, int(cfg["chunking"]["horizon"]), int(cfg["chunking"]["stride"]))
    z = dct_time(chunks.chunks)
    k_values = [int(k) for k in cfg["chunking"]["k_values"]]
    by_k, spectrum = summarize_frequency_metrics(chunks.chunks, z, k_values, data.groups)
    metrics = {
        "run": {"name": run_name, "timestamp": timestamp, "config_path": str(cfg_path.relative_to(PROJECT_ROOT)), "git_commit": _git(["rev-parse", "HEAD"])},
        "dataset": {**pusht_dataset_summary(data), "train_demos": len(train_demos), "val_demos": len(val_demos), "analysis_split": split, "analysis_demos": len(analysis_demos), "num_chunks": int(chunks.chunks.shape[0])},
        "action": {"source": "pusht_zarr", "dim_names": data.dim_names, "groups": data.groups},
        "chunking": {"horizon": int(cfg["chunking"]["horizon"]), "stride": int(cfg["chunking"]["stride"]), "k_values": k_values},
        "normalization": norm.to_jsonable(),
        "by_k": by_k,
        "spectrum": spectrum,
        "transition_conditioned": {"enabled": False, "reason": "PushT has no gripper channel"},
    }
    metrics_path = metrics_dir / f"{run_name}_{timestamp}.json"
    csv_path = metrics_dir / f"{run_name}_{timestamp}.csv"
    figure_path = figures_dir / f"{run_name}_{timestamp}.png"
    _write_json(metrics, metrics_path)
    _write_json(metrics, run_dir / "metrics.json")
    _write_csv(by_k, data.groups, csv_path)
    save_yaml(cfg, run_dir / "config.yaml")
    (run_dir / "git_status.txt").write_text(_git(["status", "--short"]), encoding="utf-8")
    plot_frequency_diagnostic(metrics, figure_path)
    print(f"Run directory: {run_dir.relative_to(PROJECT_ROOT)}")
    print(f"Metrics JSON: {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Metrics CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {figure_path.relative_to(PROJECT_ROOT)}")


def _write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_csv(rows: list[dict], groups: dict[str, list[int]], path: Path) -> None:
    fieldnames = ["k", "reconstruction_mse", "raw_smoothness", "reconstruction_smoothness", "smoothness_ratio_to_raw", "delta_action_mse", "high_energy_ratio_mean", "high_energy_ratio_median", "high_energy_ratio_std"]
    for group_name in groups:
        fieldnames.append(f"{group_name}_high_energy_ratio_mean")
        fieldnames.append(f"{group_name}_reconstruction_smoothness")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fieldnames if key in row}
            for group_name in groups:
                flat[f"{group_name}_high_energy_ratio_mean"] = row["groups"][group_name]["high_energy_ratio_mean"]
                flat[f"{group_name}_reconstruction_smoothness"] = row["groups"][group_name]["reconstruction_smoothness"]
            writer.writerow(flat)


def _git(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    main()
