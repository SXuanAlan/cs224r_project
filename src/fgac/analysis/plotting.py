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
    axes[0].axvline(8, linestyle="--", color="black", linewidth=1.0, alpha=0.6)
    axes[0].set_title("Reconstruction MSE")
    axes[0].set_xlabel("Retained DCT coefficients K")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_values, recon_smooth, marker="o", label="low-frequency recon")
    axes[1].plot(k_values, raw_smooth, linestyle="--", color="black", label="raw")
    axes[1].axvline(8, linestyle="--", color="black", linewidth=1.0, alpha=0.6)
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
    axes[2].axvline(8, linestyle="--", color="black", linewidth=1.0, alpha=0.6)
    axes[2].set_xlabel("Retained DCT coefficients K")
    axes[2].set_ylabel("Energy ratio")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(metrics["run"]["name"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_spectral_heatmaps(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Per-task heatmaps: DCT (coeff x dim), FFT (bin x dim), Morlet (scale x time)."""
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dim_names = metrics["action"]["dim_names"]
    dct = np.array(metrics["heatmap_dct"])          # [H, d]
    fft = np.array(metrics["heatmap_fft"])          # [F, d]
    tf = np.array(metrics["scalogram_tf"]["map"])   # [S, H]
    cfreqs = metrics["scalogram_tf"]["center_freqs"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    im0 = axes[0].imshow(dct, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("DCT energy (coeff x action-dim)")
    axes[0].set_xlabel("action dimension")
    axes[0].set_ylabel("DCT coefficient")
    axes[0].set_xticks(range(len(dim_names)))
    axes[0].set_xticklabels(dim_names, rotation=90, fontsize=7)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(fft, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title("FFT power (bin x action-dim)")
    axes[1].set_xlabel("action dimension")
    axes[1].set_ylabel("rfft bin")
    axes[1].set_xticks(range(len(dim_names)))
    axes[1].set_xticklabels(dim_names, rotation=90, fontsize=7)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(tf, aspect="auto", origin="lower", cmap="magma")
    axes[2].set_title("Morlet scalogram (scale x time)")
    axes[2].set_xlabel("time step in chunk")
    axes[2].set_ylabel("scale index (fine -> coarse)")
    yt = range(0, len(cfreqs), max(1, len(cfreqs) // 8))
    axes[2].set_yticks(list(yt))
    axes[2].set_yticklabels([f"{cfreqs[i]:.2f}" for i in yt], fontsize=7)
    axes[2].set_ylabel("scale center freq (cyc/sample)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{metrics['run']['name']} — spectral heatmaps")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_spectral_summary(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Three-panel per-task spectral energy figure: DCT, FFT, Morlet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    groups = metrics["action"]["groups"]
    dct = metrics["spectrum_dct"]
    fft = metrics["spectrum_fft"]
    morlet = metrics["spectrum_morlet"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    # DCT energy vs frequency index.
    dct_x = list(range(len(dct["aggregate"])))
    axes[0].plot(dct_x, dct["aggregate"], marker="o", label="aggregate")
    for group_name in groups:
        axes[0].plot(dct_x, dct["groups"][group_name], marker=".", label=group_name)
    axes[0].set_title("DCT energy spectrum")
    axes[0].set_xlabel("DCT coefficient index")
    axes[0].set_ylabel("Normalized energy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # FFT power vs normalized frequency.
    fft_x = fft["frequencies"]
    axes[1].plot(fft_x, fft["aggregate"], marker="o", label="aggregate")
    for group_name in groups:
        axes[1].plot(fft_x, fft["groups"][group_name], marker=".", label=group_name)
    axes[1].set_title("FFT power spectrum")
    axes[1].set_xlabel("Frequency (cycles/sample)")
    axes[1].set_ylabel("Normalized power")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Morlet energy vs scale center frequency.
    morlet_x = morlet["center_freqs"]
    axes[2].plot(morlet_x, morlet["aggregate"], marker="o", label="aggregate")
    for group_name in groups:
        axes[2].plot(morlet_x, morlet["groups"][group_name], marker=".", label=group_name)
    axes[2].set_title("Morlet scalogram energy")
    axes[2].set_xlabel("Scale center frequency (cycles/sample)")
    axes[2].set_ylabel("Normalized energy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(metrics["run"]["name"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
