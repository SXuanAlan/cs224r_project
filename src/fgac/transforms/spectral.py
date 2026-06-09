"""FFT and Morlet-wavelet transforms for action chunks.

These complement the DCT transforms in :mod:`fgac.transforms.dct`. All functions
operate on ``[N, H, d]`` chunks and transform along the horizon axis (axis 1),
mirroring ``dct_time``'s convention. They are pure-numpy/scipy so they run on the
CPU during expert-data analysis.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import fftconvolve


def rfft_power(actions: np.ndarray) -> np.ndarray:
    """Per-bin FFT power ``|X|**2`` along the horizon axis of ``[N, H, d]`` chunks.

    Returns an array of shape ``[N, H // 2 + 1, d]`` (real-input FFT bins).
    """
    if actions.ndim != 3:
        raise ValueError(f"Expected [N, H, d] actions, got shape {actions.shape}")
    coeffs = rfft(actions, axis=1)
    return np.abs(coeffs) ** 2


def rfft_frequencies(horizon: int) -> list[float]:
    """Normalized rfft bin frequencies (cycles/sample) for a given horizon."""
    return rfftfreq(int(horizon), d=1.0).tolist()


def _morlet_wavelet(num_points: int, scale: float, w0: float) -> np.ndarray:
    """Complex Morlet wavelet sampled on ``num_points``, centered, for a scale."""
    t = np.arange(num_points) - (num_points - 1) / 2.0
    x = t / float(scale)
    wavelet = np.pi ** -0.25 * np.exp(1j * w0 * x) * np.exp(-0.5 * x**2)
    return wavelet / np.sqrt(float(scale))


def morlet_scales(horizon: int, num_scales: int, w0: float = 6.0) -> np.ndarray:
    """Morlet scales spanning center frequencies from ~Nyquist down to ~low.

    The smallest scale is set so its center frequency ``w0 / (2*pi*scale)`` equals
    the Nyquist frequency (0.5 cycles/sample); smaller scales would be above Nyquist
    and only capture aliasing. The largest scale is the chunk length ``H``. Scale
    index 0 is the finest (highest freq); higher indices are coarser (lower freq).
    """
    min_scale = w0 / np.pi  # center freq = 0.5 (Nyquist)
    max_scale = max(min_scale + 1.0, float(horizon))
    return np.linspace(min_scale, max_scale, int(num_scales))


def morlet_scalogram(
    actions: np.ndarray,
    num_scales: int | None = None,
    w0: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-scale Morlet wavelet energy along the horizon axis of ``[N, H, d]``.

    For each scale the chunk is convolved (along time) with a complex Morlet
    wavelet, and the mean squared magnitude over time is taken as that scale's
    energy. Returns ``(energy[N, S, d], scales[S], center_freqs[S])`` where the
    center frequencies are in cycles/sample (``w0 / (2*pi*scale)``).
    """
    if actions.ndim != 3:
        raise ValueError(f"Expected [N, H, d] actions, got shape {actions.shape}")
    horizon = int(actions.shape[1])
    num_scales = int(num_scales) if num_scales else horizon
    scales = morlet_scales(horizon, num_scales, w0)

    energy = np.zeros((actions.shape[0], num_scales, actions.shape[2]), dtype=np.float64)
    for idx, scale in enumerate(scales):
        wavelet = _morlet_wavelet(horizon, scale, w0).reshape(1, horizon, 1)
        coeffs = fftconvolve(actions, wavelet, mode="same", axes=1)
        energy[:, idx, :] = np.mean(np.abs(coeffs) ** 2, axis=1)

    center_freqs = w0 / (2.0 * np.pi * scales)
    return energy, scales, center_freqs


def morlet_scalogram_tf(
    actions: np.ndarray,
    num_scales: int | None = None,
    w0: float = 6.0,
    dims: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-frequency Morlet scalogram averaged over chunks: returns ``tf[S, H]``.

    For each scale, the (channel-summed) ``|CWT|**2`` is averaged over chunks,
    giving a 2-D scale x time energy map suitable for a heatmap. Memory-light: one
    scale's convolution ``[N, H, d]`` is held at a time. Returns
    ``(tf[S, H], scales[S], center_freqs[S])``; ``tf`` is raw mean power (not yet
    normalized).
    """
    if actions.ndim != 3:
        raise ValueError(f"Expected [N, H, d] actions, got shape {actions.shape}")
    horizon = int(actions.shape[1])
    num_scales = int(num_scales) if num_scales else horizon
    scales = morlet_scales(horizon, num_scales, w0)
    sel = actions if dims is None else actions[..., dims]

    tf = np.zeros((num_scales, horizon), dtype=np.float64)
    for idx, scale in enumerate(scales):
        wavelet = _morlet_wavelet(horizon, scale, w0).reshape(1, horizon, 1)
        coeffs = fftconvolve(sel, wavelet, mode="same", axes=1)
        power = np.sum(np.abs(coeffs) ** 2, axis=-1)  # [N, H]
        tf[idx] = np.mean(power, axis=0)

    center_freqs = w0 / (2.0 * np.pi * scales)
    return tf, scales, center_freqs
