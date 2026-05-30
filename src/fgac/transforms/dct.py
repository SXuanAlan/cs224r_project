"""Temporal DCT transforms for action chunks."""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct


def dct_time(actions: np.ndarray) -> np.ndarray:
    """Apply orthonormal DCT along the horizon axis of [N, H, d] chunks."""
    if actions.ndim != 3:
        raise ValueError(f"Expected [N, H, d] actions, got shape {actions.shape}")
    return dct(actions, axis=1, norm="ortho")


def idct_time(coefficients: np.ndarray) -> np.ndarray:
    """Apply orthonormal inverse DCT along the horizon axis."""
    if coefficients.ndim != 3:
        raise ValueError(f"Expected [N, H, d] coefficients, got shape {coefficients.shape}")
    return idct(coefficients, axis=1, norm="ortho")


def low_frequency_reconstruct(actions: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct action chunks after retaining the first k DCT coefficients."""
    z = dct_time(actions)
    if not 0 < k <= z.shape[1]:
        raise ValueError(f"k must be in [1, H], got k={k}, H={z.shape[1]}")
    z_hat = np.zeros_like(z)
    z_hat[:, :k, :] = z[:, :k, :]
    return idct_time(z_hat), z


def topk_frequency_bins(coefficients: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep top-k temporal frequency bins per chunk by energy across channels.

    Args:
        coefficients: DCT coefficients with shape [N, H, d].
        k: Number of temporal frequency bins to retain. k=H returns the input.

    Returns:
        A tuple of (sparse_coefficients, selected_indices), where
        selected_indices has shape [N, k].
    """
    if coefficients.ndim != 3:
        raise ValueError(f"Expected [N, H, d] coefficients, got shape {coefficients.shape}")
    horizon = int(coefficients.shape[1])
    if not 0 < k <= horizon:
        raise ValueError(f"k must be in [1, H], got k={k}, H={horizon}")
    energy = np.sum(coefficients**2, axis=-1)
    if k == horizon:
        selected = np.broadcast_to(np.arange(horizon), (coefficients.shape[0], horizon)).copy()
    else:
        selected = np.argpartition(-energy, kth=k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(selected, np.argsort(selected, axis=1), axis=1)
    mask = np.zeros(energy.shape, dtype=bool)
    np.put_along_axis(mask, selected, True, axis=1)
    return np.where(mask[:, :, None], coefficients, 0.0), selected


def sparse_frequency_reconstruct(actions: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct chunks after retaining top-k full-spectrum DCT frequency bins."""
    z = dct_time(actions)
    z_hat, selected = topk_frequency_bins(z, k)
    return idct_time(z_hat), z, selected
