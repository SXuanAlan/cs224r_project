"""Metrics for frequency decomposition diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from fgac.transforms.dct import idct_time


def reconstruction_mse(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((reference - reconstructed) ** 2))


def smoothness(actions: np.ndarray, dims: list[int] | None = None) -> float:
    """Mean squared consecutive action difference inside chunks."""
    x = actions if dims is None else actions[..., dims]
    diff = np.diff(x, axis=1)
    return float(np.mean(np.sum(diff**2, axis=-1)))


def delta_action_mse(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    ref_delta = np.diff(reference, axis=1)
    rec_delta = np.diff(reconstructed, axis=1)
    return float(np.mean((ref_delta - rec_delta) ** 2))


def high_energy_ratio(z: np.ndarray, k: int, dims: list[int] | None = None, eps: float = 1e-12) -> np.ndarray:
    """Per-chunk high-frequency energy ratio."""
    coeffs = z if dims is None else z[..., dims]
    high = np.sum(coeffs[:, k:, :] ** 2, axis=(1, 2))
    total = np.sum(coeffs**2, axis=(1, 2))
    return high / (total + eps)


def mean_frequency_energy(z: np.ndarray, dims: list[int] | None = None, eps: float = 1e-12) -> list[float]:
    """Mean normalized energy at each temporal frequency."""
    coeffs = z if dims is None else z[..., dims]
    energy = np.sum(coeffs**2, axis=-1)
    total = np.sum(energy, axis=1, keepdims=True)
    normalized = energy / (total + eps)
    return np.mean(normalized, axis=0).tolist()


def summarize_frequency_metrics(
    actions: np.ndarray,
    z: np.ndarray,
    k_values: list[int],
    groups: dict[str, list[int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute Experiment A metrics for all retained-frequency values."""
    raw_smoothness = smoothness(actions)
    by_k: list[dict[str, Any]] = []
    for k in k_values:
        z_hat = np.zeros_like(z)
        z_hat[:, :k, :] = z[:, :k, :]
        recon = idct_time(z_hat)
        ratios = high_energy_ratio(z, k)
        row: dict[str, Any] = {
            "k": int(k),
            "reconstruction_mse": reconstruction_mse(actions, recon),
            "raw_smoothness": raw_smoothness,
            "reconstruction_smoothness": smoothness(recon),
            "smoothness_ratio_to_raw": smoothness(recon) / (raw_smoothness + 1e-12),
            "delta_action_mse": delta_action_mse(actions, recon),
            "high_energy_ratio_mean": float(np.mean(ratios)),
            "high_energy_ratio_median": float(np.median(ratios)),
            "high_energy_ratio_std": float(np.std(ratios)),
            "groups": {},
        }
        for group_name, dims in groups.items():
            group_ratios = high_energy_ratio(z, k, dims=dims)
            row["groups"][group_name] = {
                "raw_smoothness": smoothness(actions, dims=dims),
                "reconstruction_smoothness": smoothness(recon, dims=dims),
                "high_energy_ratio_mean": float(np.mean(group_ratios)),
                "high_energy_ratio_median": float(np.median(group_ratios)),
            }
        by_k.append(row)

    spectrum = {
        "aggregate": mean_frequency_energy(z),
        "groups": {group_name: mean_frequency_energy(z, dims=dims) for group_name, dims in groups.items()},
    }
    return by_k, spectrum

