"""Checkpoint loading and sampling helpers for Flow Matching policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fgac.models.flow_matching import (
    FlowMatchingMLP,
    FrequencySoftmaskTemporalUNetFlow,
    TemporalUNetFlow,
    euler_sample,
)
from fgac.transforms.dct import (
    channel_topk_frequency_bins_torch,
    dct_time_torch,
    decode_action_target,
    idct_time_torch,
    topk_frequency_bins_torch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class LoadedFlowPolicy:
    model: torch.nn.Module
    cfg: dict[str, Any]
    metadata: dict[str, Any]
    target_type: str
    model_type: str
    action_dim: int
    horizon: int
    target_shape: tuple[int, int] | None
    target_dim: int


def is_raw_anchored_residual(cfg: dict[str, Any]) -> bool:
    return cfg.get("target", {}).get("type") == "raw_anchored_residual"


def is_sparse_dct_anchored_residual(cfg: dict[str, Any]) -> bool:
    return cfg.get("target", {}).get("type") == "sparse_dct_anchored_residual"


def is_anchored_residual(cfg: dict[str, Any]) -> bool:
    return is_raw_anchored_residual(cfg) or is_sparse_dct_anchored_residual(cfg)


def residual_lambda(cfg: dict[str, Any]) -> float:
    target_cfg = cfg.get("target", {})
    return float(target_cfg.get("residual_lambda", target_cfg.get("lambda", 1.0)))


def base_noise_mode(cfg: dict[str, Any]) -> str:
    return str(cfg.get("target", {}).get("base_noise_mode", "zero"))


def base_num_flow_steps(cfg: dict[str, Any]) -> int:
    target_cfg = cfg.get("target", {})
    return int(target_cfg.get("base_num_flow_steps", cfg.get("sampling", {}).get("num_flow_steps", 32)))


def raw_base_noise_mode(cfg: dict[str, Any]) -> str:
    return base_noise_mode(cfg)


def raw_base_num_flow_steps(cfg: dict[str, Any]) -> int:
    return base_num_flow_steps(cfg)


def load_base_policy_from_config(
    cfg: dict[str, Any],
    device: torch.device,
    metadata: dict[str, Any] | None = None,
) -> LoadedFlowPolicy | None:
    if not is_anchored_residual(cfg):
        return None
    target_cfg = cfg.get("target", {})
    checkpoint = target_cfg.get("base_checkpoint")
    if not checkpoint:
        raise ValueError(f"{target_cfg.get('type')} requires target.base_checkpoint")
    require_target_type = "raw" if is_raw_anchored_residual(cfg) else str(target_cfg.get("base_target_type", "dct_sparse"))
    policy = load_flow_policy_checkpoint(checkpoint, device=device, require_target_type=require_target_type)
    if metadata is not None:
        assert_policy_compatible(policy, metadata)
    return policy


def load_raw_base_policy_from_config(
    cfg: dict[str, Any],
    device: torch.device,
    metadata: dict[str, Any] | None = None,
) -> LoadedFlowPolicy | None:
    if not is_raw_anchored_residual(cfg):
        return None
    return load_base_policy_from_config(cfg, device, metadata=metadata)


def load_flow_policy_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    require_target_type: str | None = None,
) -> LoadedFlowPolicy:
    path = _resolve(checkpoint_path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    metadata = ckpt["metadata"]
    target_cfg = cfg.get("target", {})
    target_type = str(target_cfg.get("type", "raw"))
    if require_target_type is not None and target_type != require_target_type:
        raise ValueError(f"Expected checkpoint target.type={require_target_type}, got {target_type} at {path}")

    dataset_meta = metadata.get("dataset", {})
    target_meta = metadata.get("target", {})
    action_dim = int(dataset_meta["action_dim"])
    horizon = int(target_meta.get("target_seq_len", cfg.get("chunking", {}).get("horizon", 1)))
    model_action_dim = int(target_meta.get("target_action_dim", action_dim))
    target_dim = int(target_meta.get("target_dim", horizon * model_action_dim))
    model_type = str(cfg.get("model", {}).get("type", "mlp"))
    model = _build_model(cfg, metadata, model_action_dim, horizon, target_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    target_shape = (horizon, model_action_dim) if model_type == "temporal_unet" else None
    return LoadedFlowPolicy(
        model=model,
        cfg=cfg,
        metadata=metadata,
        target_type=target_type,
        model_type=model_type,
        action_dim=action_dim,
        horizon=horizon,
        target_shape=target_shape,
        target_dim=target_dim,
    )


@torch.no_grad()
def sample_normalized_raw_actions(
    policy: LoadedFlowPolicy,
    obs: torch.Tensor,
    num_steps: int | None = None,
    noise_mode: str = "normal",
) -> torch.Tensor:
    if policy.target_type != "raw":
        raise ValueError(f"sample_normalized_raw_actions requires raw policy, got {policy.target_type}")
    steps = int(num_steps if num_steps is not None else policy.cfg.get("sampling", {}).get("num_flow_steps", 32))
    if policy.model_type == "temporal_unet":
        target = euler_sample(
            policy.model,
            obs,
            target_shape=policy.target_shape,
            num_steps=steps,
            noise_mode=noise_mode,
        )
    else:
        target = euler_sample(
            policy.model,
            obs,
            target_dim=policy.target_dim,
            num_steps=steps,
            noise_mode=noise_mode,
        )
    if target.ndim == 3:
        return target[..., : policy.action_dim]
    return target.reshape(target.shape[0], policy.horizon, policy.action_dim)


@torch.no_grad()
def sample_normalized_actions(
    policy: LoadedFlowPolicy,
    obs: torch.Tensor,
    num_steps: int | None = None,
    noise_mode: str = "normal",
) -> torch.Tensor:
    steps = int(num_steps if num_steps is not None else policy.cfg.get("sampling", {}).get("num_flow_steps", 32))
    if policy.model_type == "temporal_unet":
        target = euler_sample(
            policy.model,
            obs,
            target_shape=policy.target_shape,
            num_steps=steps,
            noise_mode=noise_mode,
        )
    else:
        target = euler_sample(
            policy.model,
            obs,
            target_dim=policy.target_dim,
            num_steps=steps,
            noise_mode=noise_mode,
        )
    return _decode_policy_target_torch(policy, target)


@torch.no_grad()
def raw_anchored_residual_target(
    action_chunk: torch.Tensor,
    obs: torch.Tensor,
    base_policy: LoadedFlowPolicy | None,
    cfg: dict[str, Any],
) -> torch.Tensor:
    if base_policy is None:
        return action_chunk
    base_actions = sample_normalized_raw_actions(
        base_policy,
        obs,
        num_steps=base_num_flow_steps(cfg),
        noise_mode=base_noise_mode(cfg),
    )
    residual = action_chunk - base_actions
    return residual


@torch.no_grad()
def sparse_dct_anchored_residual_target(
    action_chunk: torch.Tensor,
    obs: torch.Tensor,
    base_policy: LoadedFlowPolicy | None,
    cfg: dict[str, Any],
) -> torch.Tensor:
    if base_policy is None:
        residual = action_chunk
    else:
        base_actions = sample_normalized_actions(
            base_policy,
            obs,
            num_steps=base_num_flow_steps(cfg),
            noise_mode=base_noise_mode(cfg),
        )
        residual = action_chunk - base_actions
    return dct_time_torch(residual)


@torch.no_grad()
def combine_raw_base_and_residual(
    residual: torch.Tensor,
    obs: torch.Tensor,
    base_policy: LoadedFlowPolicy | None,
    cfg: dict[str, Any],
) -> torch.Tensor:
    if base_policy is None:
        return residual
    base_actions = sample_normalized_raw_actions(
        base_policy,
        obs,
        num_steps=base_num_flow_steps(cfg),
        noise_mode=base_noise_mode(cfg),
    )
    return base_actions + residual_lambda(cfg) * residual


@torch.no_grad()
def combine_base_and_residual(
    residual: torch.Tensor,
    obs: torch.Tensor,
    base_policy: LoadedFlowPolicy | None,
    cfg: dict[str, Any],
) -> torch.Tensor:
    if base_policy is None:
        return residual
    base_actions = sample_normalized_actions(
        base_policy,
        obs,
        num_steps=base_num_flow_steps(cfg),
        noise_mode=base_noise_mode(cfg),
    )
    return base_actions + residual_lambda(cfg) * residual


def assert_policy_compatible(policy: LoadedFlowPolicy, metadata: dict[str, Any]) -> None:
    dataset_meta = metadata.get("dataset", {})
    target_meta = metadata.get("target", {})
    obs_dim = int(dataset_meta.get("obs_dim", -1))
    policy_obs_dim = int(policy.metadata.get("dataset", {}).get("obs_dim", -2))
    action_dim = int(dataset_meta.get("action_dim", -1))
    horizon = int(target_meta.get("target_seq_len", metadata.get("chunking", {}).get("horizon", policy.horizon)))
    if policy_obs_dim != obs_dim:
        raise ValueError(f"Base policy obs_dim={policy_obs_dim} is incompatible with residual obs_dim={obs_dim}")
    if policy.action_dim != action_dim:
        raise ValueError(f"Base policy action_dim={policy.action_dim} is incompatible with residual action_dim={action_dim}")
    if policy.horizon != horizon:
        raise ValueError(f"Base policy horizon={policy.horizon} is incompatible with residual horizon={horizon}")


def _decode_policy_target_torch(policy: LoadedFlowPolicy, target: torch.Tensor) -> torch.Tensor:
    target_type = policy.target_type
    target_cfg = policy.cfg.get("target", {})
    horizon = policy.horizon
    action_dim = policy.action_dim
    if target_type in {"raw", "raw_anchored_residual"}:
        if target.ndim == 3:
            return target[..., :action_dim]
        return target.reshape(target.shape[0], horizon, action_dim)

    if target.ndim == 2:
        model_action_dim = target.shape[-1] // horizon
        seq = target.reshape(target.shape[0], horizon, model_action_dim)
    else:
        seq = target

    if target_type == "dct_lowfreq":
        k = int(target_cfg["dct_k"])
        coeffs = torch.zeros((target.shape[0], horizon, action_dim), device=target.device, dtype=target.dtype)
        coeffs[:, :k, :] = seq[..., :action_dim]
        return idct_time_torch(coeffs)

    if target_type in {"dct_fullfreq", "dct_softmask", "sparse_dct_anchored_residual"}:
        return idct_time_torch(seq[..., :action_dim])

    if target_type == "dct_sparse":
        coeffs, _ = topk_frequency_bins_torch(seq[..., :action_dim], int(target_cfg["dct_k"]))
        return idct_time_torch(coeffs)

    if target_type == "dct_channel_sparse":
        coeffs, _ = channel_topk_frequency_bins_torch(seq[..., :action_dim], target_cfg["dct_k"])
        return idct_time_torch(coeffs)

    if target_type == "dct_sparse_residual":
        if seq.shape[-1] < action_dim * 2:
            raise ValueError(f"dct_sparse_residual needs 2d channels, got {seq.shape[-1]} for d={action_dim}")
        coeffs, _ = topk_frequency_bins_torch(seq[..., :action_dim], int(target_cfg["dct_k"]))
        residual = seq[..., action_dim : action_dim * 2]
        return idct_time_torch(coeffs) + residual

    decoded = decode_action_target(
        target.detach().cpu().numpy(),
        target_type=target_type,
        horizon=horizon,
        action_dim=action_dim,
        dct_k=target_cfg.get("dct_k"),
    )
    return torch.from_numpy(decoded).to(device=target.device, dtype=target.dtype)


def _build_model(
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    model_action_dim: int,
    target_seq_len: int,
    target_dim: int,
) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "mlp"))
    obs_dim = int(metadata["dataset"]["obs_dim"])
    if model_type == "mlp":
        return FlowMatchingMLP(
            obs_dim=obs_dim,
            target_dim=target_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            time_embed_dim=int(model_cfg.get("time_embed_dim", 64)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if model_type != "temporal_unet":
        raise ValueError(f"Unsupported model.type: {model_type}")
    base_model = TemporalUNetFlow(
        obs_dim=obs_dim,
        action_dim=model_action_dim,
        base_dim=int(model_cfg.get("base_dim", 128)),
        dim_mults=tuple(int(v) for v in model_cfg.get("dim_mults", [1, 2, 4])),
        time_embed_dim=int(model_cfg.get("time_embed_dim", 128)),
        cond_dim=int(model_cfg.get("cond_dim", 256)),
        kernel_size=int(model_cfg.get("kernel_size", 5)),
        groups=int(model_cfg.get("groups", 8)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    target_type = cfg.get("target", {}).get("type")
    if target_type == "dct_softmask":
        gate_cfg = cfg.get("target", {}).get("gate", {})
        return FrequencySoftmaskTemporalUNetFlow(
            base_model,
            sequence_length=target_seq_len,
            init_logit=float(gate_cfg.get("init_logit", 2.0)),
            temperature=float(gate_cfg.get("temperature", 1.0)),
        )
    if target_type == "sparse_dct_anchored_residual":
        gate_cfg = cfg.get("target", {}).get("gate", {})
        gate_mode = str(gate_cfg.get("mode", gate_cfg.get("type", "softmask")))
        if gate_mode in {"softmask", "soft_gate", "soft"}:
            return FrequencySoftmaskTemporalUNetFlow(
                base_model,
                sequence_length=target_seq_len,
                init_logit=float(gate_cfg.get("init_logit", -2.0)),
                temperature=float(gate_cfg.get("temperature", 1.0)),
            )
    return base_model


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
