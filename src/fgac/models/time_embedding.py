"""Time embedding modules."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar Flow Matching time t in [0, 1]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = t * freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

