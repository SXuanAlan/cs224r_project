#!/usr/bin/env python
"""Train PushT raw or DCT temporal-UNet Flow Matching baselines."""

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
from scipy.fft import idct
from torch.utils.data import DataLoader

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fgac.analysis.frequency_metrics import delta_action_mse, reconstruction_mse, smoothness
from fgac.data.chunk_dataset import FlowMatchingChunkDataset
from fgac.data.normalization import fit_minmax, fit_zscore
from fgac.data.pusht_zarr import load_pusht_replay, pusht_dataset_summary
from fgac.data.robomimic_hdf5 import build_obs_action_chunks, split_demos
from fgac.models.flow_matching import TemporalUNetFlow, euler_sample, flow_matching_loss
from fgac.transforms.dct import topk_frequency_bins
from fgac.utils.config import load_yaml, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
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
    for path in [run_dir, ckpt_dir, metrics_dir, videos_dir]:
        path.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "config.yaml")
    (run_dir / "git_status.txt").write_text(_git(["status", "--short"]), encoding="utf-8")

    dataset_path = PROJECT_ROOT / cfg["dataset"]["path"]
    data = load_pusht_replay(
        dataset_path,
        obs_key=cfg["dataset"].get("obs_key", "state"),
        action_key=cfg["dataset"].get("action_key", "action"),
        limit_demos=cfg["dataset"].get("limit_demos"),
    )
    train_demos, val_demos = split_demos(data.demos, float(cfg["dataset"]["val_fraction"]), int(cfg["run"]["seed"]))
    action_stats = fit_minmax([data.actions_by_demo[d] for d in train_demos], eps=float(cfg["normalization"]["action"]["eps"]))
    obs_stats = fit_zscore([data.obs_by_demo[d] for d in train_demos], eps=float(cfg["normalization"]["obs"]["eps"]))
    norm_actions = {demo: action_stats.normalize(actions) for demo, actions in data.actions_by_demo.items()}
    norm_obs = {demo: obs_stats.normalize(obs) for demo, obs in data.obs_by_demo.items()}

    horizon = int(cfg["chunking"]["horizon"])
    stride = int(cfg["chunking"]["stride"])
    observation_horizon = int(cfg["chunking"].get("observation_horizon", 2))
    train_chunks = build_obs_action_chunks(norm_obs, norm_actions, train_demos, horizon, stride, observation_horizon)
    val_chunks = build_obs_action_chunks(norm_obs, norm_actions, val_demos, horizon, stride, observation_horizon)
    train_ds = FlowMatchingChunkDataset(train_chunks.obs, train_chunks.action_chunks, cfg["target"]["type"], cfg["target"].get("dct_k"))
    val_ds = FlowMatchingChunkDataset(val_chunks.obs, val_chunks.action_chunks, cfg["target"]["type"], cfg["target"].get("dct_k"))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=int(cfg["training"]["num_workers"]))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=int(cfg["training"]["num_workers"]))

    device = torch.device("cuda" if cfg["training"]["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    model = TemporalUNetFlow(
        obs_dim=train_ds.obs.shape[-1],
        action_dim=train_ds.target_action_dim,
        base_dim=int(cfg["model"]["base_dim"]),
        dim_mults=tuple(int(v) for v in cfg["model"]["dim_mults"]),
        time_embed_dim=int(cfg["model"]["time_embed_dim"]),
        cond_dim=int(cfg["model"]["cond_dim"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        groups=int(cfg["model"]["groups"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    max_steps = _max_steps(cfg)
    steps_per_epoch = len(train_loader) if max_steps is None else min(len(train_loader), max_steps)
    scheduler = _build_scheduler(optimizer, cfg, max(1, steps_per_epoch * int(cfg["training"]["epochs"])))
    ema = EMAModel(model, float(cfg["training"].get("ema", {}).get("decay", 0.995))) if cfg["training"].get("ema", {}).get("enabled", True) else None
    wandb_run = _init_wandb(cfg, run_name, timestamp, run_dir)

    metadata = {
        "run": {"name": run_name, "timestamp": timestamp, "config_path": str(cfg_path.relative_to(PROJECT_ROOT))},
        "dataset": {
            **pusht_dataset_summary(data),
            "train_demos": len(train_demos),
            "val_demos": len(val_demos),
            "train_chunks": len(train_ds),
            "val_chunks": len(val_ds),
            "obs_dim": int(train_ds.obs.shape[-1]),
            "observation_horizon": observation_horizon,
            "obs_key": cfg["dataset"].get("obs_key", "state"),
        },
        "action": {"source": "pusht_zarr", "dim_names": data.dim_names, "groups": data.groups},
        "target": {**cfg["target"], "target_seq_len": train_ds.target_seq_len, "target_action_dim": train_ds.target_action_dim, "target_dim": train_ds.target_dim},
        "model": cfg["model"],
        "normalization": {"action": action_stats.to_jsonable(), "obs": obs_stats.to_jsonable()},
    }
    _write_json(metadata, run_dir / "metadata.json")

    best_val = float("inf")
    history: list[dict[str, Any]] = []
    eval_every = int(cfg["training"].get("eval_every_epochs", 10))
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_loss = _train_one_epoch(model, optimizer, train_loader, device, cfg, max_steps, scheduler, ema)
        should_eval = epoch == 1 or epoch % eval_every == 0 or epoch == int(cfg["training"]["epochs"])
        row: dict[str, Any] = {"epoch": epoch, "train/fm_loss": train_loss, "train/lr": float(optimizer.param_groups[0]["lr"])}
        val_metrics = None
        if should_eval:
            with use_ema_weights(model, ema):
                val_metrics = _evaluate(model, val_loader, device, cfg, train_ds)
            row.update({f"val/{k}": v for k, v in val_metrics.items()})
        history.append(row)
        if wandb_run is not None:
            wandb_run.log(row, step=epoch)
        if val_metrics is None:
            print(f"epoch {epoch:03d} train_loss={train_loss:.6f} lr={row['train/lr']:.3e}")
        else:
            print(f"epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_metrics['fm_loss']:.6f} val_mse={val_metrics['action_mse']:.6f} val_smooth={val_metrics['smoothness']:.6f}")
        if val_metrics is not None and val_metrics["action_mse"] < best_val:
            best_val = val_metrics["action_mse"]
            _save_checkpoint(model, optimizer, cfg, metadata, ckpt_dir / "best.pt", epoch, val_metrics, ema)

    final_metrics = {"metadata": metadata, "history": history, "best_val_action_mse": best_val, "run_dir": str(run_dir.relative_to(PROJECT_ROOT)), "checkpoint": str((ckpt_dir / "best.pt").relative_to(PROJECT_ROOT))}
    if cfg.get("simulation_video", {}).get("enabled", True):
        sim_path = videos_dir / "simulation_rollout.mp4"
        sim_result = _run_pusht_eval(ckpt_dir / "best.pt", sim_path, cfg)
        final_metrics["simulation_video"] = str(sim_path.relative_to(PROJECT_ROOT))
        final_metrics["simulation_video_result"] = sim_result
    metrics_path = metrics_dir / f"{run_name}_{timestamp}.json"
    _write_json(final_metrics, metrics_path)
    _write_json(final_metrics, run_dir / "metrics.json")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Run directory: {run_dir.relative_to(PROJECT_ROOT)}")
    print(f"Best checkpoint: {(ckpt_dir / 'best.pt').relative_to(PROJECT_ROOT)}")
    print(f"Metrics JSON: {metrics_path.relative_to(PROJECT_ROOT)}")


def _train_one_epoch(model, optimizer, loader, device, cfg, max_steps, scheduler, ema) -> float:
    model.train()
    losses = []
    for step, batch in enumerate(loader, start=1):
        obs = batch["obs"].to(device)
        target = _target_from_batch(batch, cfg).to(device)
        loss = flow_matching_loss(model, obs, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
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
def _evaluate(model, loader, device, cfg, train_ds) -> dict[str, float]:
    model.eval()
    fm_losses, preds, trues = [], [], []
    for batch in loader:
        obs = batch["obs"].to(device)
        target = _target_from_batch(batch, cfg).to(device)
        fm_losses.append(float(flow_matching_loss(model, obs, target).detach().cpu()))
        pred = euler_sample(model, obs, target_shape=train_ds.target_shape, num_steps=int(cfg["sampling"]["num_flow_steps"]))
        preds.append(_decode_actions(pred.cpu().numpy(), cfg, train_ds))
        trues.append(batch["action_chunk"].numpy())
    pred_actions = np.concatenate(preds, axis=0)
    true_actions = np.concatenate(trues, axis=0)
    return {
        "fm_loss": float(np.mean(fm_losses)),
        "action_mse": reconstruction_mse(true_actions, pred_actions),
        "smoothness": smoothness(pred_actions),
        "true_smoothness": smoothness(true_actions),
        "delta_action_mse": delta_action_mse(true_actions, pred_actions),
    }


def _target_from_batch(batch, cfg):
    return batch["target_seq"]


def _decode_actions(target: np.ndarray, cfg: dict[str, Any], train_ds) -> np.ndarray:
    if cfg["target"]["type"] == "raw":
        return target
    if cfg["target"]["type"] == "dct_lowfreq":
        coeffs = np.zeros((target.shape[0], train_ds.horizon, train_ds.action_dim), dtype=np.float32)
        coeffs[:, : int(cfg["target"]["dct_k"]), :] = target
        return idct(coeffs, axis=1, norm="ortho")
    if cfg["target"]["type"] == "dct_fullfreq":
        return idct(target, axis=1, norm="ortho")
    if cfg["target"]["type"] == "dct_sparse":
        coeffs, _ = topk_frequency_bins(target, int(cfg["target"]["dct_k"]))
        return idct(coeffs, axis=1, norm="ortho")
    raise ValueError(f"Unsupported target.type: {cfg['target']['type']}")


def _max_steps(cfg):
    value = cfg["training"].get("max_train_steps_per_epoch")
    return int(value) if value is not None else None


def _build_scheduler(optimizer, cfg, total_steps: int):
    sched = cfg["training"].get("lr_scheduler", {})
    if sched.get("type", "none") in {None, "none"}:
        return None
    warmup_steps = int(sched.get("warmup_steps", 0))
    min_lr_scale = float(sched.get("min_lr_scale", 0.0))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMAModel:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {name: value.detach().clone() for name, value in self.shadow.items()}


@contextmanager
def use_ema_weights(model, ema):
    if ema is None:
        yield
        return
    backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
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


def _save_checkpoint(model, optimizer, cfg, metadata, path: Path, epoch: int, val_metrics: dict[str, float], ema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if ema is not None:
        state = {k: ema.shadow[k].detach().clone() if k in ema.shadow else v.detach().clone() for k, v in state.items()}
    torch.save({"model_state_dict": state, "raw_model_state_dict": model.state_dict(), "ema_state_dict": ema.state_dict() if ema else None, "optimizer_state_dict": optimizer.state_dict(), "config": cfg, "metadata": metadata, "epoch": epoch, "val_metrics": val_metrics}, path)


def _run_pusht_eval(checkpoint_path: Path, output_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    sim = cfg.get("simulation_video", {})
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "eval_pusht_rollout.py"), "--checkpoint", str(checkpoint_path.relative_to(PROJECT_ROOT)), "--output", str(output_path.relative_to(PROJECT_ROOT)), "--n-rollouts", str(int(sim.get("n_rollouts", 50))), "--horizon", str(int(sim.get("horizon", 300))), "--seed", str(int(sim.get("seed", cfg["run"].get("seed", 0))))]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    log_path = output_path.with_suffix(".log")
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    metrics_path = output_path.with_suffix(".json")
    return {"command": cmd, "returncode": result.returncode, "log_path": str(log_path.relative_to(PROJECT_ROOT)), "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)) if metrics_path.exists() else None, "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-2000:]}


def _init_wandb(cfg, run_name: str, timestamp: str, run_dir: Path):
    if not cfg.get("wandb", {}).get("enabled", False):
        return None
    wandb_cfg = cfg["wandb"]
    if wandb_cfg.get("mode"):
        os.environ["WANDB_MODE"] = str(wandb_cfg["mode"])
    import wandb

    return wandb.init(project=wandb_cfg.get("project", "cs224r-fgac"), entity=wandb_cfg.get("entity"), name=f"{run_name}_{timestamp}", mode=wandb_cfg.get("mode", "online"), tags=wandb_cfg.get("tags", []), config=cfg, dir=str(run_dir))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _git(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    main()
