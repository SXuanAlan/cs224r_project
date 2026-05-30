#!/usr/bin/env python
"""Aggregate Can sparse-K sweep rollout metrics and plot success vs smoothness."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "p3_can_k_sweep"


SPARSE_PATTERNS = {
    4: "outputs/eval/can_ph_fm_dct_sparse_k4_*/env/metrics.json",
    8: "outputs/eval/can_ph_fm_dct_sparse_k8_*/env/metrics.json",
    12: "outputs/eval/can_ph_fm_dct_sparse_k12_*/env/metrics.json",
    16: "outputs/eval/can_ph_fm_raw_*/env/metrics.json",
}
LOWFREQ_PATTERN = "outputs/eval/can_ph_fm_dct_lowfreq_k8_*/env/metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR.relative_to(PROJECT_ROOT)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    metrics_path = output_dir / "metrics" / "k_sweep.json"
    figure_path = output_dir / "figures" / "can_smoothness_vs_success.png"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    sparse = [_load_k(k, pattern) for k, pattern in SPARSE_PATTERNS.items()]
    lowfreq = _load_point("lowfreq_k8", LOWFREQ_PATTERN)
    payload = {
        "run": {
            "name": "p3_can_k_sweep",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "note": "K=16 uses the existing raw FM run as specified in the prompt.",
        },
        "sparse": sparse,
        "lowfreq_k8": lowfreq,
    }
    _write_json(payload, metrics_path)
    _plot(payload, figure_path)

    print(f"Metrics: {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {figure_path.relative_to(PROJECT_ROOT)}")
    for point in sparse:
        print(
            f"K={point['k']}: smooth={point['smoothness']:.6f}, "
            f"success={point['success_rate']:.3f}, mse={point['val_action_mse']:.6f}"
        )
    print(
        "Low-Freq K=8: "
        f"smooth={lowfreq['smoothness']:.6f}, success={lowfreq['success_rate']:.3f}"
    )


def _load_k(k: int, pattern: str) -> dict[str, Any]:
    point = _load_point(f"sparse_k{k}", pattern)
    point["k"] = k
    if k == 16:
        point["method"] = "raw_fm"
    else:
        point["method"] = "dct_sparse"
    return point


def _load_point(name: str, pattern: str) -> dict[str, Any]:
    matches = sorted(PROJECT_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    if name == "sparse_k8":
        matches = [path for path in matches if "_seed" not in path.parents[1].name]
    if not matches:
        raise FileNotFoundError(f"No metrics found for {name}: {pattern}")
    path = matches[-1]
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    val = data["best_validation"]
    sim = data["simulation"]
    return {
        "name": name,
        "run_id": data["run_id"],
        "metrics_path": _rel(path),
        "checkpoint": data.get("checkpoint"),
        "epoch": val.get("epoch"),
        "val_action_mse": val.get("action_mse"),
        "smoothness": val.get("smoothness"),
        "delta_action_mse": val.get("delta_action_mse"),
        "success_rate": sim.get("success_rate"),
        "num_rollouts": sim.get("num_rollouts"),
        "num_successes": sim.get("num_successes"),
    }


def _plot(payload: dict[str, Any], path: Path) -> None:
    sparse = sorted(payload["sparse"], key=lambda point: point["k"])
    lowfreq = payload["lowfreq_k8"]
    xs = [point["smoothness"] for point in sparse]
    ys = [point["success_rate"] for point in sparse]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(xs, ys, marker="o", color="#2f6f8f", label="Sparse full-spectrum")
    for point in sparse:
        ax.annotate(
            f"K={point['k']}",
            (point["smoothness"], point["success_rate"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    ax.scatter(
        [lowfreq["smoothness"]],
        [lowfreq["success_rate"]],
        marker="X",
        s=90,
        color="#b84a4a",
        label="Low-Freq K=8",
        zorder=3,
    )
    ax.annotate(
        "Low-Freq K=8",
        (lowfreq["smoothness"], lowfreq["success_rate"]),
        textcoords="offset points",
        xytext=(6, -14),
        fontsize=9,
    )
    ax.set_xlabel("Within-chunk smoothness")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Can K Sweep: Smoothness vs Success")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
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
