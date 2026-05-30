"""Metrics for frequency decomposition diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from fgac.transforms.dct import dct_time, idct_time


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


def per_phase_spectrum(
    actions_per_chunk: np.ndarray,
    gripper_dim_index: int,
    tau: float = 1.0,
    channel_groups: dict[str, list[int]] | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Compute transition-conditioned DCT spectra for action chunks.

    Chunks are labeled transition if the gripper action changes by more than
    ``tau`` between any adjacent timestep. Energy is normalized per chunk
    before averaging, so each returned vector sums to approximately 1.
    """
    if actions_per_chunk.ndim != 3:
        raise ValueError(f"Expected [N, H, d] actions, got shape {actions_per_chunk.shape}")
    if not 0 <= gripper_dim_index < actions_per_chunk.shape[-1]:
        raise ValueError(f"Invalid gripper_dim_index={gripper_dim_index} for dim={actions_per_chunk.shape[-1]}")

    groups = channel_groups or _default_channel_groups(actions_per_chunk.shape[-1], gripper_dim_index)
    gripper = actions_per_chunk[..., gripper_dim_index]
    transition = np.any(np.abs(np.diff(gripper, axis=1)) > float(tau), axis=1)
    z = dct_time(actions_per_chunk)

    result: dict[str, Any] = {
        "tau": float(tau),
        "gripper_dim_index": int(gripper_dim_index),
        "num_chunks": int(actions_per_chunk.shape[0]),
        "transition_chunks": int(np.sum(transition)),
        "non_transition_chunks": int(np.sum(~transition)),
        "phases": {},
    }
    for name, mask in {"transition": transition, "non_transition": ~transition}.items():
        result["phases"][name] = {
            "aggregate": _phase_energy(z[mask], dims=None, eps=eps),
            "per_channel": {
                group_name: _phase_energy(z[mask], dims=dims, eps=eps)
                for group_name, dims in groups.items()
            },
        }
    return result


def _phase_energy(z: np.ndarray, dims: list[int] | None, eps: float) -> list[float]:
    if z.shape[0] == 0:
        horizon = int(z.shape[1]) if z.ndim == 3 else 0
        return [0.0 for _ in range(horizon)]
    coeffs = z if dims is None else z[..., dims]
    energy = np.sum(coeffs**2, axis=-1)
    total = np.sum(energy, axis=1, keepdims=True)
    normalized = energy / (total + eps)
    return np.mean(normalized, axis=0).tolist()


def _default_channel_groups(action_dim: int, gripper_dim_index: int) -> dict[str, list[int]]:
    if action_dim >= 7 and gripper_dim_index == 6:
        return {
            "translation": [0, 1, 2],
            "rotation": [3, 4, 5],
            "gripper": [6],
        }
    non_gripper = [idx for idx in range(action_dim) if idx != gripper_dim_index]
    return {
        "translation": non_gripper,
        "rotation": [],
        "gripper": [gripper_dim_index],
    }


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
