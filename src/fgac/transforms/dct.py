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

