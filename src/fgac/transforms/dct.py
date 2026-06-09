"""Temporal DCT transforms for action chunks."""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected for training, optional for pure numpy analysis.
    torch = None


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


def topk_frequency_bins_torch(coefficients: "torch.Tensor", k: int) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Torch equivalent of topk_frequency_bins for [B, H, d] coefficients."""
    if torch is None:
        raise ImportError("topk_frequency_bins_torch requires torch.")
    if coefficients.ndim != 3:
        raise ValueError(f"Expected [B, H, d] coefficients, got shape {tuple(coefficients.shape)}")
    horizon = int(coefficients.shape[1])
    if not 0 < k <= horizon:
        raise ValueError(f"k must be in [1, H], got k={k}, H={horizon}")
    energy = torch.sum(coefficients**2, dim=-1)
    if k == horizon:
        selected = torch.arange(horizon, device=coefficients.device).expand(coefficients.shape[0], horizon)
    else:
        selected = torch.topk(energy, k=k, dim=1, largest=True, sorted=False).indices
        selected = torch.sort(selected, dim=1).values
    mask = torch.zeros_like(energy, dtype=torch.bool)
    mask.scatter_(1, selected, True)
    return torch.where(mask.unsqueeze(-1), coefficients, torch.zeros_like(coefficients)), selected


def _normalize_channel_k(channel_k, action_dim: int, horizon: int) -> np.ndarray:
    if channel_k is None:
        raise ValueError("channel-wise DCT targets require dct_k")
    if np.isscalar(channel_k):
        ks = np.full(action_dim, int(channel_k), dtype=np.int64)
    else:
        ks = np.asarray(channel_k, dtype=np.int64)
        if ks.ndim != 1:
            raise ValueError(f"dct_k must be a scalar or 1-D list, got shape {ks.shape}")
        if int(ks.size) != action_dim:
            raise ValueError(f"dct_k length must match action_dim={action_dim}, got {int(ks.size)}")
    if np.any(ks <= 0) or np.any(ks > horizon):
        raise ValueError(f"all dct_k values must be in [1, H={horizon}], got {ks.tolist()}")
    return ks


def channel_topk_frequency_bins(coefficients: np.ndarray, channel_k) -> tuple[np.ndarray, np.ndarray]:
    """Keep top-k temporal bins independently for each action channel.

    This is the channel-aware analogue of ``topk_frequency_bins``. A scalar
    ``channel_k`` reproduces per-channel top-k selection with the same budget
    for every channel; a list allocates different bandwidths per channel.
    """
    if coefficients.ndim != 3:
        raise ValueError(f"Expected [N, H, d] coefficients, got shape {coefficients.shape}")
    horizon = int(coefficients.shape[1])
    action_dim = int(coefficients.shape[2])
    ks = _normalize_channel_k(channel_k, action_dim, horizon)
    energy = coefficients**2
    mask = np.zeros(coefficients.shape, dtype=bool)
    rows = np.arange(coefficients.shape[0])[:, None]
    for channel, k in enumerate(ks.tolist()):
        if k == horizon:
            mask[:, :, channel] = True
            continue
        selected = np.argpartition(-energy[:, :, channel], kth=k - 1, axis=1)[:, :k]
        mask[rows, selected, channel] = True
    return np.where(mask, coefficients, 0.0), mask


def channel_topk_frequency_bins_torch(coefficients: "torch.Tensor", channel_k) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Torch equivalent of channel_topk_frequency_bins for [B, H, d] coefficients."""
    if torch is None:
        raise ImportError("channel_topk_frequency_bins_torch requires torch.")
    if coefficients.ndim != 3:
        raise ValueError(f"Expected [B, H, d] coefficients, got shape {tuple(coefficients.shape)}")
    horizon = int(coefficients.shape[1])
    action_dim = int(coefficients.shape[2])
    ks = _normalize_channel_k(channel_k, action_dim, horizon)
    energy = coefficients.square()
    mask = torch.zeros_like(coefficients, dtype=torch.bool)
    for channel, k in enumerate(ks.tolist()):
        if k == horizon:
            mask[:, :, channel] = True
            continue
        selected = torch.topk(energy[:, :, channel], k=k, dim=1, largest=True, sorted=False).indices
        mask[:, :, channel].scatter_(1, selected, True)
    return torch.where(mask, coefficients, torch.zeros_like(coefficients)), mask


def sparse_frequency_reconstruct(actions: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct chunks after retaining top-k full-spectrum DCT frequency bins."""
    z = dct_time(actions)
    z_hat, selected = topk_frequency_bins(z, k)
    return idct_time(z_hat), z, selected


def sparse_residual_target(actions: np.ndarray, k: int) -> np.ndarray:
    """Build a sparse-DCT plus time-domain residual target.

    The first d channels are sparse DCT coefficients. The second d channels are
    the residual between the normalized action chunk and the sparse-DCT
    reconstruction. Decoding is additive.
    """
    if actions.ndim != 3:
        raise ValueError(f"Expected [N, H, d] actions, got shape {actions.shape}")
    z = dct_time(actions)
    z_sparse, _ = topk_frequency_bins(z, k)
    residual = actions - idct_time(z_sparse)
    return np.concatenate([z_sparse, residual], axis=-1).astype(np.float32)


def idct_time_torch(coefficients: "torch.Tensor") -> "torch.Tensor":
    """Differentiable orthonormal inverse DCT along axis 1 for [N, H, d]."""
    if torch is None:
        raise ImportError("idct_time_torch requires torch.")
    if coefficients.ndim != 3:
        raise ValueError(f"Expected [N, H, d] coefficients, got shape {tuple(coefficients.shape)}")
    horizon = int(coefficients.shape[1])
    basis = _orthonormal_dct_basis(horizon, coefficients.device, coefficients.dtype)
    return torch.einsum("kh,bkd->bhd", basis, coefficients)


def dct_time_torch(actions: "torch.Tensor") -> "torch.Tensor":
    """Differentiable orthonormal DCT along axis 1 for [N, H, d]."""
    if torch is None:
        raise ImportError("dct_time_torch requires torch.")
    if actions.ndim != 3:
        raise ValueError(f"Expected [N, H, d] actions, got shape {tuple(actions.shape)}")
    horizon = int(actions.shape[1])
    basis = _orthonormal_dct_basis(horizon, actions.device, actions.dtype)
    return torch.einsum("kh,bhd->bkd", basis, actions)


def _orthonormal_dct_basis(horizon: int, device: "torch.device", dtype: "torch.dtype") -> "torch.Tensor":
    n = torch.arange(horizon, device=device, dtype=dtype)
    k = torch.arange(horizon, device=device, dtype=dtype).unsqueeze(1)
    basis = torch.cos(np.pi * (n + 0.5) * k / float(horizon))
    basis[0, :] *= np.sqrt(1.0 / float(horizon))
    if horizon > 1:
        basis[1:, :] *= np.sqrt(2.0 / float(horizon))
    return basis


def decode_action_target(
    target: np.ndarray,
    target_type: str,
    horizon: int,
    action_dim: int,
    dct_k: int | None = None,
) -> np.ndarray:
    """Decode a model target sequence back to normalized time-domain actions."""
    horizon = int(horizon)
    action_dim = int(action_dim)
    if target_type in {"raw", "raw_anchored_residual"}:
        if target.ndim == 3:
            return target[..., :action_dim]
        return target.reshape(target.shape[0], horizon, action_dim)

    if target_type == "dct_lowfreq":
        if dct_k is None:
            raise ValueError("dct_lowfreq requires dct_k")
        k = int(dct_k)
        coeffs = np.zeros((target.shape[0], horizon, action_dim), dtype=np.float32)
        coeff_target = target if target.ndim == 3 else target.reshape(target.shape[0], k, action_dim)
        coeffs[:, :k, :] = coeff_target
        return idct_time(coeffs)

    if target_type == "dct_fullfreq":
        coeffs = target if target.ndim == 3 else target.reshape(target.shape[0], horizon, action_dim)
        return idct_time(coeffs[..., :action_dim])

    if target_type == "dct_sparse":
        if dct_k is None:
            raise ValueError("dct_sparse requires dct_k")
        coeffs = target if target.ndim == 3 else target.reshape(target.shape[0], horizon, action_dim)
        coeffs, _ = topk_frequency_bins(coeffs[..., :action_dim], int(dct_k))
        return idct_time(coeffs)

    if target_type == "dct_channel_sparse":
        if dct_k is None:
            raise ValueError("dct_channel_sparse requires dct_k")
        coeffs = target if target.ndim == 3 else target.reshape(target.shape[0], horizon, action_dim)
        coeffs, _ = channel_topk_frequency_bins(coeffs[..., :action_dim], dct_k)
        return idct_time(coeffs)

    if target_type == "dct_sparse_residual":
        if dct_k is None:
            raise ValueError("dct_sparse_residual requires dct_k")
        seq = target if target.ndim == 3 else target.reshape(target.shape[0], horizon, action_dim * 2)
        if seq.shape[-1] < action_dim * 2:
            raise ValueError(f"dct_sparse_residual needs 2d channels, got {seq.shape[-1]} for d={action_dim}")
        coeffs = seq[..., :action_dim]
        residual = seq[..., action_dim : action_dim * 2]
        coeffs, _ = topk_frequency_bins(coeffs, int(dct_k))
        return idct_time(coeffs) + residual

    if target_type in {"dct_softmask", "sparse_dct_anchored_residual"}:
        coeffs = target if target.ndim == 3 else target.reshape(target.shape[0], horizon, action_dim)
        return idct_time(coeffs[..., :action_dim])

    raise ValueError(f"Unsupported target_type: {target_type}")
