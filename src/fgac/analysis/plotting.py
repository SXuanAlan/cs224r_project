"""Plotting helpers for frequency diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_frequency_diagnostic(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Create a compact Experiment A diagnostic figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = metrics["by_k"]
    k_values = [row["k"] for row in rows]
    rec_mse = [row["reconstruction_mse"] for row in rows]
    raw_smooth = [row["raw_smoothness"] for row in rows]
    recon_smooth = [row["reconstruction_smoothness"] for row in rows]
    high_energy = [row["high_energy_ratio_mean"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(k_values, rec_mse, marker="o")
    axes[0].set_title("Reconstruction MSE")
    axes[0].set_xlabel("Retained DCT coefficients K")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_values, recon_smooth, marker="o", label="low-frequency recon")
    axes[1].plot(k_values, raw_smooth, linestyle="--", color="black", label="raw")
    axes[1].set_title("Within-Chunk Smoothness")
    axes[1].set_xlabel("Retained DCT coefficients K")
    axes[1].set_ylabel("Mean squared action delta")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(k_values, high_energy, marker="o", label="aggregate")
    groups = metrics["action"]["groups"]
    for group_name in groups:
        axes[2].plot(
            k_values,
            [row["groups"][group_name]["high_energy_ratio_mean"] for row in rows],
            marker=".",
            label=group_name,
        )
    axes[2].set_title("High-Frequency Energy Ratio")
    axes[2].set_xlabel("Retained DCT coefficients K")
    axes[2].set_ylabel("Energy ratio")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(metrics["run"]["name"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

