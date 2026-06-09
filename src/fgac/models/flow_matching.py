"""Flow Matching models for action chunks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from fgac.models.time_embedding import SinusoidalTimeEmbedding


class FlowMatchingMLP(nn.Module):
    """Predict velocity v(x_t, t, obs) for rectified Flow Matching."""

    def __init__(
        self,
        obs_dim: int,
        target_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        time_embed_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        input_dim = obs_dim + target_dim + time_embed_dim
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, target_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x_t, t_emb, obs], dim=-1))


def flow_matching_loss(model: nn.Module, obs: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Rectified-flow objective with linear interpolation from Gaussian x0 to data x1."""
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, device=x1.device, dtype=x1.dtype)
    t_view = t.reshape(x1.shape[0], *([1] * (x1.ndim - 1)))
    x_t = (1.0 - t_view) * x0 + t_view * x1
    target_v = x1 - x0
    pred_v = model(x_t, t, obs)
    return torch.mean((pred_v - target_v) ** 2)


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    obs: torch.Tensor,
    num_steps: int,
    target_dim: int | None = None,
    target_shape: Sequence[int] | None = None,
    noise_mode: str = "normal",
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample x_1 by integrating dx/dt = v_theta(x, t, obs) from Gaussian noise."""
    if target_shape is None:
        if target_dim is None:
            raise ValueError("Either target_shape or target_dim must be provided.")
        sample_shape = (obs.shape[0], int(target_dim))
    else:
        sample_shape = (obs.shape[0], *[int(v) for v in target_shape])
    if initial_noise is not None:
        if tuple(initial_noise.shape) != tuple(sample_shape):
            raise ValueError(f"initial_noise shape {tuple(initial_noise.shape)} does not match {sample_shape}")
        x = initial_noise.to(device=obs.device, dtype=obs.dtype)
    elif noise_mode == "normal":
        x = torch.randn(sample_shape, device=obs.device, dtype=obs.dtype)
    elif noise_mode == "zero":
        x = torch.zeros(sample_shape, device=obs.device, dtype=obs.dtype)
    else:
        raise ValueError(f"Unsupported noise_mode: {noise_mode}")
    dt = 1.0 / float(num_steps)
    for step in range(num_steps):
        t_value = torch.full((obs.shape[0], 1), step / float(num_steps), device=obs.device, dtype=obs.dtype)
        x = x + dt * model(x, t_value, obs)
    return x


class ResidualBlock1D(nn.Module):
    """1D residual block with FiLM conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(_valid_groups(groups, in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(_valid_groups(groups, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_channels * 2))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale_shift = self.cond(cond)[:, :, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1.0 + scale) + shift
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.residual(x)


class TemporalUNetFlow(nn.Module):
    """Temporal 1D UNet velocity model for action chunk Flow Matching.

    Inputs and outputs are shaped [batch, sequence, action_dim], where sequence
    is either the raw action horizon H or the retained DCT coefficient count K.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        base_dim: int = 128,
        dim_mults: Sequence[int] = (1, 2, 4),
        time_embed_dim: int = 128,
        cond_dim: int = 256,
        kernel_size: int = 5,
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(dim_mults) < 2:
            raise ValueError("TemporalUNetFlow needs at least two dim_mults for a UNet.")
        dims = [int(base_dim * mult) for mult in dim_mults]
        self.action_dim = int(action_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.cond_net = nn.Sequential(
            nn.Linear(time_embed_dim + obs_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.init_conv = nn.Conv1d(action_dim, dims[0], kernel_size=1)

        self.downs = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.downs.append(
                nn.ModuleDict(
                    {
                        "block1": ResidualBlock1D(dims[i], dims[i], cond_dim, kernel_size, groups, dropout),
                        "block2": ResidualBlock1D(dims[i], dims[i], cond_dim, kernel_size, groups, dropout),
                        "down": nn.Conv1d(dims[i], dims[i + 1], kernel_size=4, stride=2, padding=1),
                    }
                )
            )

        self.mid1 = ResidualBlock1D(dims[-1], dims[-1], cond_dim, kernel_size, groups, dropout)
        self.mid2 = ResidualBlock1D(dims[-1], dims[-1], cond_dim, kernel_size, groups, dropout)

        self.ups = nn.ModuleList()
        for i in reversed(range(len(dims) - 1)):
            self.ups.append(
                nn.ModuleDict(
                    {
                        "up": nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv1d(dims[i + 1], dims[i], kernel_size=3, padding=1),
                        ),
                        "block1": ResidualBlock1D(dims[i] * 2, dims[i], cond_dim, kernel_size, groups, dropout),
                        "block2": ResidualBlock1D(dims[i], dims[i], cond_dim, kernel_size, groups, dropout),
                    }
                )
            )

        self.final_block = ResidualBlock1D(dims[0], dims[0], cond_dim, kernel_size, groups, dropout)
        self.final_conv = nn.Conv1d(dims[0], action_dim, kernel_size=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 3:
            raise ValueError(f"TemporalUNetFlow expects [B, L, D], got {tuple(x_t.shape)}")
        if x_t.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {x_t.shape[-1]}")
        cond = self.cond_net(torch.cat([self.time_embed(t), obs], dim=-1))
        x = self.init_conv(x_t.transpose(1, 2))
        skips: list[torch.Tensor] = []
        for down in self.downs:
            x = down["block1"](x, cond)
            x = down["block2"](x, cond)
            skips.append(x)
            x = down["down"](x)
        x = self.mid1(x, cond)
        x = self.mid2(x, cond)
        for up in self.ups:
            x = up["up"](x)
            skip = skips.pop()
            x = _align_length(x, skip.shape[-1])
            x = torch.cat([x, skip], dim=1)
            x = up["block1"](x, cond)
            x = up["block2"](x, cond)
        x = self.final_block(x, cond)
        return self.final_conv(x).transpose(1, 2)


class FrequencySoftmaskTemporalUNetFlow(nn.Module):
    """Temporal UNet with learned per-frequency target masks.

    The gates define the DCT target distribution used during training, so
    sampled coefficients are decoded directly at evaluation time.
    """

    def __init__(
        self,
        base_model: TemporalUNetFlow,
        sequence_length: int,
        init_logit: float = 2.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.action_dim = base_model.action_dim
        self.sequence_length = int(sequence_length)
        self.temperature = float(temperature)
        self.gate_logits = nn.Parameter(torch.full((self.sequence_length,), float(init_logit)))

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits / max(self.temperature, 1.0e-6))

    def gate_l1(self) -> torch.Tensor:
        return self.gates().mean()

    def effective_k(self) -> torch.Tensor:
        return self.gates().sum()

    def transform_target(self, target: torch.Tensor) -> torch.Tensor:
        return target * self.gates().reshape(1, self.sequence_length, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.base_model(x_t, t, obs)


def _align_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] > target_len:
        return x[..., :target_len]
    return F.pad(x, (0, target_len - x.shape[-1]))


def _valid_groups(requested: int, channels: int) -> int:
    requested = max(1, min(int(requested), int(channels)))
    for groups in range(requested, 0, -1):
        if channels % groups == 0:
            return groups
    return 1
