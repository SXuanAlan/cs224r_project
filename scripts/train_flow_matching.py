#!/usr/bin/env python
"""Train raw or DCT action chunk Flow Matching baselines."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fgac.analysis.frequency_metrics import delta_action_mse, reconstruction_mse, smoothness
from fgac.analysis.eval_video import save_action_chunk_eval_video
from fgac.data.chunk_dataset import FlowMatchingChunkDataset
from fgac.data.normalization import fit_minmax, fit_zscore
from fgac.data.robomimic_hdf5 import (
    build_obs_action_chunks,
    dataset_summary,
    load_actions,
    load_observations,
    split_demos,
)
from fgac.models.flow_matching import (
    FlowMatchingMLP,
    FrequencySoftmaskTemporalUNetFlow,
    TemporalUNetFlow,
    euler_sample,
    flow_matching_loss,
)
from fgac.models.policy_io import (
    base_noise_mode,
    base_num_flow_steps,
    combine_base_and_residual,
    combine_raw_base_and_residual,
    is_anchored_residual,
    is_raw_anchored_residual,
    is_sparse_dct_anchored_residual,
    load_base_policy_from_config,
    raw_anchored_residual_target,
    sample_normalized_actions,
    sparse_dct_anchored_residual_target,
)
from fgac.transforms.dct import dct_time_torch, decode_action_target, idct_time_torch
from fgac.utils.config import load_yaml, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    cfg = load_yaml(cfg_path)
    _seed_everything(int(cfg["run"]["seed"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg["run"]["name"]
    run_dir = PROJECT_ROOT / cfg["outputs"]["log_dir"] / f"{run_name}_{timestamp}"
    ckpt_dir = PROJECT_ROOT / cfg["outputs"]["checkpoints_dir"] / f"{run_name}_{timestamp}"
    metrics_dir = PROJECT_ROOT / cfg["outputs"]["metrics_dir"]
    videos_dir = PROJECT_ROOT / cfg["outputs"].get("videos_dir", "outputs/videos") / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "config.yaml")
    (run_dir / "git_status.txt").write_text(_git(["status", "--short"]), encoding="utf-8")

    dataset_path = PROJECT_ROOT / cfg["dataset"]["path"]
    action_data = load_actions(
        dataset_path,
        source=cfg["action"]["source"],
        action_dict_keys=cfg["action"]["action_dict_keys"],
        legacy_key=cfg["action"]["legacy_key"],
        limit_demos=cfg["dataset"].get("limit_demos"),
    )
    train_demos, val_demos = split_demos(
        action_data.demos,
        val_fraction=float(cfg["dataset"]["val_fraction"]),
        seed=int(cfg["run"]["seed"]),
    )
    obs_by_demo, obs_dim_names = load_observations(dataset_path, action_data.demos, cfg["dataset"]["obs_keys"])
    action_stats = fit_minmax(
        [action_data.actions_by_demo[demo] for demo in train_demos],
        eps=float(cfg["normalization"]["action"]["eps"]),
    )
    obs_stats = fit_zscore(
        [obs_by_demo[demo] for demo in train_demos],
        eps=float(cfg["normalization"]["obs"]["eps"]),
    )
    norm_actions = {demo: action_stats.normalize(actions) for demo, actions in action_data.actions_by_demo.items()}
    norm_obs = {demo: obs_stats.normalize(obs) for demo, obs in obs_by_demo.items()}

    horizon = int(cfg["chunking"]["horizon"])
    stride = int(cfg["chunking"]["stride"])
    observation_horizon = int(cfg["chunking"].get("observation_horizon", 1))
    train_chunks = build_obs_action_chunks(
        norm_obs,
        norm_actions,
        train_demos,
        horizon=horizon,
        stride=stride,
        observation_horizon=observation_horizon,
    )
    val_chunks = build_obs_action_chunks(
        norm_obs,
        norm_actions,
        val_demos,
        horizon=horizon,
        stride=stride,
        observation_horizon=observation_horizon,
    )

    target_type = cfg["target"]["type"]
    dct_k = cfg["target"].get("dct_k")
    train_ds = FlowMatchingChunkDataset(train_chunks.obs, train_chunks.action_chunks, target_type, dct_k=dct_k)
    val_ds = FlowMatchingChunkDataset(val_chunks.obs, val_chunks.action_chunks, target_type, dct_k=dct_k)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        drop_last=False,
    )

    requested_device = cfg["training"]["device"]
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    model = _build_model(cfg, train_ds).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    steps_per_epoch = len(train_loader) if max_steps_from_cfg(cfg) is None else min(len(train_loader), max_steps_from_cfg(cfg))
    scheduler = _build_scheduler(optimizer, cfg, total_steps=max(1, steps_per_epoch * int(cfg["training"]["epochs"])))
    ema = EMAModel(model, decay=float(cfg["training"].get("ema", {}).get("decay", 0.995))) if cfg["training"].get("ema", {}).get("enabled", False) else None

    wandb_run = _init_wandb(cfg, run_name, timestamp, run_dir)
    metadata = {
        "run": {"name": run_name, "timestamp": timestamp, "config_path": str(cfg_path.relative_to(PROJECT_ROOT))},
        "dataset": {
            **dataset_summary(action_data),
            "train_demos": len(train_demos),
            "val_demos": len(val_demos),
            "train_chunks": len(train_ds),
            "val_chunks": len(val_ds),
            "obs_frame_dim": len(obs_dim_names),
            "obs_dim": int(train_ds.obs.shape[-1]),
            "observation_horizon": observation_horizon,
            "obs_keys": cfg["dataset"]["obs_keys"],
        },
        "action": {
            "source": action_data.source,
            "dim_names": action_data.dim_names,
            "groups": action_data.groups,
        },
        "target": {
            **cfg["target"],
            "target_seq_len": train_ds.target_seq_len,
            "target_action_dim": train_ds.target_action_dim,
            "target_dim": train_ds.target_dim,
        },
        "model": cfg.get("model", {}),
        "normalization": {
            "action": action_stats.to_jsonable(),
            "obs": obs_stats.to_jsonable(),
        },
    }
    _write_json(metadata, run_dir / "metadata.json")
    base_policy = load_base_policy_from_config(cfg, device, metadata=metadata)
    _attach_base_action_chunks(train_ds, base_policy, cfg, device)
    _attach_base_action_chunks(val_ds, base_policy, cfg, device)

    best_val = float("inf")
    history: list[dict[str, Any]] = []
    epochs = int(cfg["training"]["epochs"])
    eval_every_epochs = int(cfg["training"].get("eval_every_epochs", 1))
    max_steps = max_steps_from_cfg(cfg)
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            cfg,
            max_steps=max_steps,
            scheduler=scheduler,
            ema=ema,
            base_policy=base_policy,
        )
        should_eval = epoch == 1 or epoch % eval_every_epochs == 0 or epoch == epochs
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train/fm_loss": train_loss,
            "train/lr": current_lr,
        }
        val_metrics = None
        if should_eval:
            with use_ema_weights(model, ema):
                val_metrics = _evaluate(model, val_loader, device, cfg, train_ds, base_policy=base_policy)
            row.update({f"val/{k}": v for k, v in val_metrics.items()})
        history.append(row)
        if wandb_run is not None:
            wandb_run.log(row, step=epoch)
        if val_metrics is None:
            print(f"epoch {epoch:03d} train_loss={train_loss:.6f} lr={current_lr:.3e}")
        else:
            print(
                f"epoch {epoch:03d} train_loss={train_loss:.6f} "
                f"val_loss={val_metrics['fm_loss']:.6f} val_mse={val_metrics['action_mse']:.6f} "
                f"val_smooth={val_metrics['smoothness']:.6f}"
            )
        if val_metrics is not None and val_metrics["action_mse"] < best_val:
            best_val = val_metrics["action_mse"]
            _save_checkpoint(model, optimizer, cfg, metadata, ckpt_dir / "best.pt", epoch, val_metrics, ema=ema)

    final_metrics = {
        "metadata": metadata,
        "history": history,
        "best_val_action_mse": best_val,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "checkpoint": str((ckpt_dir / "best.pt").relative_to(PROJECT_ROOT)),
    }
    video_path = None
    if cfg.get("eval_video", {}).get("enabled", False):
        video_path = videos_dir / "validation_action_chunks.mp4"
        video_payload = _collect_eval_video_payload(model, val_loader, device, cfg, train_ds, base_policy=base_policy)
        save_action_chunk_eval_video(
            true_actions=video_payload["true"],
            pred_actions=video_payload["pred"],
            output_path=video_path,
            dim_names=action_data.dim_names,
            num_examples=int(cfg["eval_video"].get("num_examples", 4)),
            fps=int(cfg["eval_video"].get("fps", 4)),
        )
        final_metrics["eval_video"] = str(video_path.relative_to(PROJECT_ROOT))
        if wandb_run is not None and cfg["eval_video"].get("log_to_wandb", True):
            import wandb

            wandb_run.log({"eval/validation_action_chunks_video": wandb.Video(str(video_path), format="mp4")})

    sim_video_path = None
    if cfg.get("simulation_video", {}).get("enabled", False):
        sim_video_path = videos_dir / "simulation_rollout.mp4"
        sim_result = _run_simulation_video_eval(
            checkpoint_path=ckpt_dir / "best.pt",
            output_path=sim_video_path,
            cfg=cfg,
        )
        final_metrics["simulation_video"] = str(sim_video_path.relative_to(PROJECT_ROOT))
        final_metrics["simulation_video_result"] = sim_result
        if wandb_run is not None and sim_video_path.exists():
            import wandb

            wandb_run.log({"eval/simulation_rollout_video": wandb.Video(str(sim_video_path), format="mp4")})

    metrics_path = metrics_dir / f"{run_name}_{timestamp}.json"
    _write_json(final_metrics, metrics_path)
    _write_json(final_metrics, run_dir / "metrics.json")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Run directory: {run_dir.relative_to(PROJECT_ROOT)}")
    print(f"Best checkpoint: {(ckpt_dir / 'best.pt').relative_to(PROJECT_ROOT)}")
    print(f"Metrics JSON: {metrics_path.relative_to(PROJECT_ROOT)}")
    if video_path is not None:
        print(f"Eval video: {video_path.relative_to(PROJECT_ROOT)}")
    if sim_video_path is not None:
        print(f"Simulation video: {sim_video_path.relative_to(PROJECT_ROOT)}")


def _build_model(cfg: dict[str, Any], train_ds: FlowMatchingChunkDataset) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "mlp")
    if model_type == "mlp":
        return FlowMatchingMLP(
            obs_dim=train_ds.obs.shape[-1],
            target_dim=train_ds.target_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            time_embed_dim=int(model_cfg.get("time_embed_dim", 64)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if model_type == "temporal_unet":
        base_model = TemporalUNetFlow(
            obs_dim=train_ds.obs.shape[-1],
            action_dim=train_ds.target_action_dim,
            base_dim=int(model_cfg.get("base_dim", 128)),
            dim_mults=tuple(int(v) for v in model_cfg.get("dim_mults", [1, 2, 4])),
            time_embed_dim=int(model_cfg.get("time_embed_dim", 128)),
            cond_dim=int(model_cfg.get("cond_dim", 256)),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            groups=int(model_cfg.get("groups", 8)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        if cfg["target"]["type"] == "dct_softmask":
            gate_cfg = cfg["target"].get("gate", {})
            return FrequencySoftmaskTemporalUNetFlow(
                base_model,
                sequence_length=train_ds.target_seq_len,
                init_logit=float(gate_cfg.get("init_logit", 2.0)),
                temperature=float(gate_cfg.get("temperature", 1.0)),
            )
        if cfg["target"]["type"] == "sparse_dct_anchored_residual":
            gate_cfg = cfg["target"].get("gate", {})
            gate_mode = str(gate_cfg.get("mode", gate_cfg.get("type", "softmask")))
            if gate_mode in {"softmask", "soft_gate", "soft"}:
                return FrequencySoftmaskTemporalUNetFlow(
                    base_model,
                    sequence_length=train_ds.target_seq_len,
                    init_logit=float(gate_cfg.get("init_logit", -2.0)),
                    temperature=float(gate_cfg.get("temperature", 1.0)),
                )
        return base_model
    raise ValueError(f"Unsupported model.type: {model_type}")


def _target_from_batch(batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> torch.Tensor:
    if cfg.get("model", {}).get("type", "mlp") == "temporal_unet":
        return batch["target_seq"]
    return batch["target"]


def _sample_targets(model, obs: torch.Tensor, cfg: dict[str, Any], train_ds: FlowMatchingChunkDataset) -> torch.Tensor:
    if cfg.get("model", {}).get("type", "mlp") == "temporal_unet":
        return euler_sample(
            model,
            obs,
            target_shape=train_ds.target_shape,
            num_steps=int(cfg["sampling"]["num_flow_steps"]),
        )
    return euler_sample(
        model,
        obs,
        target_dim=train_ds.target_dim,
        num_steps=int(cfg["sampling"]["num_flow_steps"]),
    )


@torch.no_grad()
def _attach_base_action_chunks(
    dataset: FlowMatchingChunkDataset,
    base_policy,
    cfg: dict[str, Any],
    device: torch.device,
) -> None:
    if base_policy is None or not is_anchored_residual(cfg):
        return
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    chunks: list[np.ndarray] = []
    for batch in loader:
        obs = batch["obs"].to(device)
        base_actions = sample_normalized_actions(
            base_policy,
            obs,
            num_steps=base_num_flow_steps(cfg),
            noise_mode=base_noise_mode(cfg),
        )
        chunks.append(base_actions.detach().cpu().numpy())
    dataset.set_base_action_chunks(np.concatenate(chunks, axis=0))


def max_steps_from_cfg(cfg: dict[str, Any]) -> int | None:
    max_steps = cfg["training"].get("max_train_steps_per_epoch")
    return int(max_steps) if max_steps is not None else None


def _build_scheduler(optimizer, cfg: dict[str, Any], total_steps: int):
    scheduler_cfg = cfg["training"].get("lr_scheduler", {})
    if scheduler_cfg.get("type", "none") in {None, "none"}:
        return None
    if scheduler_cfg["type"] != "cosine":
        raise ValueError(f"Unsupported lr_scheduler.type: {scheduler_cfg['type']}")
    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
    min_lr_scale = float(scheduler_cfg.get("min_lr_scale", 0.0))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMAModel:
    """Exponential moving average of trainable model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.shadow.items()}


@contextmanager
def use_ema_weights(model: torch.nn.Module, ema: EMAModel | None):
    if ema is None:
        yield
        return
    backup = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    try:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(ema.shadow[name])
        yield
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(backup[name])


def _train_one_epoch(
    model,
    optimizer,
    loader,
    device,
    cfg,
    max_steps: int | None,
    scheduler=None,
    ema: "EMAModel | None" = None,
    base_policy=None,
) -> float:
    model.train()
    losses: list[float] = []
    grad_clip = float(cfg["training"]["grad_clip_norm"])
    for step, batch in enumerate(loader, start=1):
        obs = batch["obs"].to(device)
        target = _target_from_batch(batch, cfg).to(device)
        action_chunk = batch["action_chunk"].to(device)
        base_action_chunk = batch.get("base_action_chunk")
        base_action_chunk = base_action_chunk.to(device) if base_action_chunk is not None else None
        loss = _training_loss(
            model,
            obs,
            target,
            cfg,
            action_chunk=action_chunk,
            base_policy=base_policy,
            base_action_chunk=base_action_chunk,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        losses.append(float(loss.detach().cpu()))
        if max_steps is not None and step >= max_steps:
            break
    return float(np.mean(losses))


@torch.no_grad()
def _evaluate(model, loader, device, cfg, train_ds: FlowMatchingChunkDataset, base_policy=None) -> dict[str, float]:
    model.eval()
    fm_losses: list[float] = []
    pred_actions: list[np.ndarray] = []
    true_actions: list[np.ndarray] = []
    for batch in loader:
        obs = batch["obs"].to(device)
        target = _target_from_batch(batch, cfg).to(device)
        action_chunk = batch["action_chunk"].to(device)
        base_action_chunk = batch.get("base_action_chunk")
        base_action_chunk = base_action_chunk.to(device) if base_action_chunk is not None else None
        fm_losses.append(
            float(
                _training_loss(
                    model,
                    obs,
                    target,
                    cfg,
                    action_chunk=action_chunk,
                    base_policy=base_policy,
                    base_action_chunk=base_action_chunk,
                )
                .detach()
                .cpu()
            )
        )
        pred_target = _sample_targets(model, obs, cfg, train_ds)
        pred_actions.append(_decode_actions(pred_target.cpu().numpy(), cfg, train_ds, model=model, obs=obs, base_policy=base_policy))
        true_actions.append(batch["action_chunk"].numpy())
    pred = np.concatenate(pred_actions, axis=0)
    true = np.concatenate(true_actions, axis=0)
    metrics = {
        "fm_loss": float(np.mean(fm_losses)),
        "action_mse": reconstruction_mse(true, pred),
        "smoothness": smoothness(pred),
        "true_smoothness": smoothness(true),
        "delta_action_mse": delta_action_mse(true, pred),
    }
    metrics.update(_gate_metrics(model))
    return metrics


def _training_loss(
    model,
    obs: torch.Tensor,
    target: torch.Tensor,
    cfg: dict[str, Any],
    action_chunk: torch.Tensor | None = None,
    base_policy=None,
    base_action_chunk: torch.Tensor | None = None,
) -> torch.Tensor:
    if is_raw_anchored_residual(cfg):
        if action_chunk is None:
            raise ValueError("raw_anchored_residual training requires action_chunk")
        target = action_chunk - base_action_chunk if base_action_chunk is not None else raw_anchored_residual_target(action_chunk, obs, base_policy, cfg)
    elif is_sparse_dct_anchored_residual(cfg):
        if action_chunk is None:
            raise ValueError("sparse_dct_anchored_residual training requires action_chunk")
        if base_action_chunk is not None:
            target = dct_time_torch(action_chunk - base_action_chunk)
        else:
            target = sparse_dct_anchored_residual_target(action_chunk, obs, base_policy, cfg)
    flow_target = model.transform_target(target) if hasattr(model, "transform_target") else target
    loss = flow_matching_loss(model, obs, flow_target)
    gate_cfg = cfg["target"].get("gate", {})
    reconstruction_weight = float(gate_cfg.get("reconstruction_weight", 0.0))
    if cfg["target"]["type"] == "dct_softmask" and reconstruction_weight > 0.0 and action_chunk is not None:
        recon = idct_time_torch(flow_target[..., : action_chunk.shape[-1]])
        loss = loss + reconstruction_weight * torch.mean((recon - action_chunk) ** 2)
    if is_sparse_dct_anchored_residual(cfg) and reconstruction_weight > 0.0 and action_chunk is not None:
        recon = idct_time_torch(flow_target[..., : action_chunk.shape[-1]])
        target_residual = idct_time_torch(target[..., : action_chunk.shape[-1]])
        loss = loss + reconstruction_weight * torch.mean((recon - target_residual) ** 2)
    sparsity_weight = float(gate_cfg.get("sparsity_weight", 0.0))
    if sparsity_weight > 0.0 and hasattr(model, "gate_l1"):
        loss = loss + sparsity_weight * model.gate_l1()
    return loss


def _decode_actions(
    target: np.ndarray,
    cfg: dict[str, Any],
    train_ds: FlowMatchingChunkDataset,
    model=None,
    obs: torch.Tensor | None = None,
    base_policy=None,
) -> np.ndarray:
    decoded = decode_action_target(
        target,
        target_type=cfg["target"]["type"],
        horizon=train_ds.horizon,
        action_dim=train_ds.action_dim,
        dct_k=cfg["target"].get("dct_k"),
    )
    if is_raw_anchored_residual(cfg):
        if obs is None:
            raise ValueError("raw_anchored_residual decoding requires obs")
        residual = torch.from_numpy(decoded).to(device=obs.device, dtype=obs.dtype)
        return combine_raw_base_and_residual(residual, obs, base_policy, cfg).detach().cpu().numpy()
    if is_sparse_dct_anchored_residual(cfg):
        if obs is None:
            raise ValueError("sparse_dct_anchored_residual decoding requires obs")
        residual = torch.from_numpy(decoded).to(device=obs.device, dtype=obs.dtype)
        return combine_base_and_residual(residual, obs, base_policy, cfg).detach().cpu().numpy()
    if is_anchored_residual(cfg):
        raise ValueError(f"Unsupported anchored residual target: {cfg['target']['type']}")
    return decoded

def _gate_metrics(model) -> dict[str, float]:
    if not hasattr(model, "gates"):
        return {}
    gates = model.gates().detach().cpu().numpy().astype(np.float64)
    return {
        "gate_mean": float(np.mean(gates)),
        "gate_min": float(np.min(gates)),
        "gate_max": float(np.max(gates)),
        "effective_k": float(np.sum(gates)),
    }


@torch.no_grad()
def _collect_eval_video_payload(model, loader, device, cfg, train_ds: FlowMatchingChunkDataset, base_policy=None) -> dict[str, np.ndarray]:
    model.eval()
    batch = next(iter(loader))
    obs = batch["obs"].to(device)
    pred_target = _sample_targets(model, obs, cfg, train_ds)
    return {
        "pred": _decode_actions(pred_target.cpu().numpy(), cfg, train_ds, model=model, obs=obs, base_policy=base_policy),
        "true": batch["action_chunk"].numpy(),
    }


def _init_wandb(cfg: dict[str, Any], run_name: str, timestamp: str, run_dir: Path):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    if wandb_cfg.get("mode"):
        os.environ["WANDB_MODE"] = str(wandb_cfg["mode"])
    import wandb

    kwargs = {
        "project": wandb_cfg.get("project", "cs224r-fgac"),
        "name": f"{run_name}_{timestamp}",
        "mode": wandb_cfg.get("mode", "offline"),
        "tags": wandb_cfg.get("tags", []),
        "config": cfg,
        "dir": str(run_dir),
    }
    if wandb_cfg.get("entity"):
        kwargs["entity"] = wandb_cfg["entity"]
    return wandb.init(**kwargs)


def _run_simulation_video_eval(checkpoint_path: Path, output_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    sim_cfg = cfg.get("simulation_video", {})
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_sim_rollout.py"),
        "--checkpoint",
        str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "--output",
        str(output_path.relative_to(PROJECT_ROOT)),
        "--horizon",
        str(int(sim_cfg.get("horizon", 120))),
        "--n-rollouts",
        str(int(sim_cfg.get("n_rollouts", 1))),
        "--reset-mode",
        str(sim_cfg.get("reset_mode", "dataset")),
        "--height",
        str(int(sim_cfg.get("height", 512))),
        "--width",
        str(int(sim_cfg.get("width", 512))),
        "--seed",
        str(int(sim_cfg.get("seed", cfg["run"].get("seed", 0)))),
    ]
    camera_names = sim_cfg.get("camera_names", ["agentview"])
    if camera_names:
        command.extend(["--camera-names", *camera_names])
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".log")
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    metrics_path = output_path.with_suffix(".json")
    rollout_metrics = None
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            rollout_metrics = json.load(f)
    return {
        "command": command,
        "returncode": result.returncode,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)) if metrics_path.exists() else None,
        "metrics": rollout_metrics,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def _save_checkpoint(
    model,
    optimizer,
    cfg,
    metadata,
    path: Path,
    epoch: int,
    val_metrics: dict[str, float],
    ema: EMAModel | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.state_dict()
    if ema is not None:
        model_state = {
            key: ema.shadow[key].detach().clone() if key in ema.shadow else value.detach().clone()
            for key, value in model_state.items()
        }
    torch.save(
        {
            "model_state_dict": model_state,
            "raw_model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "metadata": metadata,
            "epoch": epoch,
            "val_metrics": val_metrics,
        },
        path,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip()


if __name__ == "__main__":
    main()
