#!/usr/bin/env python
"""Evaluate a trained ManiSkill Flow Matching policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import imageio.v2 as imageio
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fgac.models.flow_matching import (
    FrequencySoftmaskTemporalUNetFlow,
    TemporalUNetFlow,
    euler_sample,
)
from fgac.models.policy_io import (
    combine_base_and_residual,
    combine_raw_base_and_residual,
    is_anchored_residual,
    is_raw_anchored_residual,
    is_sparse_dct_anchored_residual,
    load_base_policy_from_config,
)
from fgac.transforms.dct import decode_action_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-rollouts", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--residual-lambda", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = _resolve(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    metadata = ckpt["metadata"]
    if args.residual_lambda is not None:
        cfg.setdefault("target", {})["residual_lambda"] = float(args.residual_lambda)
    sim = cfg.get("simulation_video", {})
    device_name = args.device or cfg["training"].get("device", "cuda")
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = _load_model(ckpt, metadata, cfg, device)

    n_rollouts = int(args.n_rollouts or sim.get("n_rollouts", 20))
    max_videos = int(args.max_videos if args.max_videos is not None else sim.get("max_videos", n_rollouts))
    horizon = int(args.horizon or sim.get("horizon", 200))
    seed = int(args.seed if args.seed is not None else sim.get("seed", cfg["run"].get("seed", 0)))
    output_base = _resolve(args.output) if args.output else _default_output_path(ckpt_path, metadata)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for idx in range(n_rollouts):
        output_path = output_base if idx < max_videos else None
        if output_path is not None and n_rollouts > 1:
            output_path = output_base.with_name(f"{output_base.stem}_env_{idx:03d}{output_base.suffix}")
        result = _rollout_one(model, cfg, metadata, output_path, horizon, seed + idx, device)
        results.append(result)
        if output_path is not None:
            print(f"Saved ManiSkill video: {output_path.relative_to(PROJECT_ROOT)}")
        else:
            print("Skipped ManiSkill video save for this rollout")
        print(f"  rollout={idx} return={result['return']:.4f} success={result['success']}")

    payload = {
        "checkpoint": str(ckpt_path),
        "task": metadata["dataset"].get("env_id", "maniskill"),
        "reset_mode": "env",
        "horizon": horizon,
        "action_exec_horizon": int(sim.get("action_exec_horizon", 8)),
        "max_videos": max_videos,
        "seed": seed,
        "residual_lambda": cfg.get("target", {}).get("residual_lambda") if is_anchored_residual(cfg) else None,
        "summary": _summarize(results),
        "rollouts": results,
    }
    metrics_path = output_base.with_suffix(".json")
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"ManiSkill rollout metrics: {metrics_path.relative_to(PROJECT_ROOT)}")


def _load_model(ckpt: dict[str, Any], metadata: dict[str, Any], cfg: dict[str, Any], device: torch.device):
    target_meta = metadata["target"]
    model_action_dim = int(target_meta.get("target_action_dim", metadata["dataset"]["action_dim"]))
    target_seq_len = int(target_meta["target_seq_len"])
    model_cfg = cfg["model"]
    base_model = TemporalUNetFlow(
        obs_dim=int(metadata["dataset"]["obs_dim"]),
        action_dim=model_action_dim,
        base_dim=int(model_cfg["base_dim"]),
        dim_mults=tuple(int(v) for v in model_cfg["dim_mults"]),
        time_embed_dim=int(model_cfg["time_embed_dim"]),
        cond_dim=int(model_cfg["cond_dim"]),
        kernel_size=int(model_cfg["kernel_size"]),
        groups=int(model_cfg["groups"]),
        dropout=float(model_cfg["dropout"]),
    )
    if cfg["target"]["type"] == "dct_softmask":
        gate_cfg = cfg["target"].get("gate", {})
        model = FrequencySoftmaskTemporalUNetFlow(
            base_model,
            sequence_length=target_seq_len,
            init_logit=float(gate_cfg.get("init_logit", 2.0)),
            temperature=float(gate_cfg.get("temperature", 1.0)),
        ).to(device)
    elif cfg["target"]["type"] == "sparse_dct_anchored_residual":
        gate_cfg = cfg["target"].get("gate", {})
        gate_mode = str(gate_cfg.get("mode", gate_cfg.get("type", "softmask")))
        if gate_mode in {"softmask", "soft_gate", "soft"}:
            model = FrequencySoftmaskTemporalUNetFlow(
                base_model,
                sequence_length=target_seq_len,
                init_logit=float(gate_cfg.get("init_logit", -2.0)),
                temperature=float(gate_cfg.get("temperature", 1.0)),
            ).to(device)
        else:
            model = base_model.to(device)
    else:
        model = base_model.to(device)
    model.target_shape = (target_seq_len, model_action_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.base_policy = load_base_policy_from_config(cfg, device, metadata=metadata)
    model.raw_base_policy = model.base_policy
    return model


def _rollout_one(
    model,
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    output_path: Path | None,
    horizon: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    dataset_cfg = cfg["dataset"]
    sim = cfg.get("simulation_video", {})
    env = gym.make(
        dataset_cfg["env_id"],
        obs_mode=dataset_cfg.get("env_obs_mode", dataset_cfg.get("obs_mode", "state")),
        control_mode=dataset_cfg.get("control_mode", "pd_ee_delta_pose"),
        render_mode="rgb_array",
        reward_mode=dataset_cfg.get("reward_mode", "sparse"),
        max_episode_steps=horizon,
    )
    obs, _ = env.reset(seed=seed)
    obs_vec = _flatten_obs(obs)
    observation_horizon = int(metadata["dataset"].get("observation_horizon", 2))
    obs_history = [obs_vec for _ in range(observation_horizon)]
    action_buffer: list[np.ndarray] = []
    buffer_index = 0
    total_reward = 0.0
    success = False
    video_skip = int(sim.get("video_skip", 2))
    fps = int(sim.get("fps", 20))
    action_exec_horizon = int(sim.get("action_exec_horizon", 8))
    writer = imageio.get_writer(output_path, fps=fps, macro_block_size=8) if output_path is not None else None

    try:
        for step in range(horizon):
            if step % max(video_skip, 1) == 0 and writer is not None:
                writer.append_data(_render_frame(env))
            if buffer_index >= len(action_buffer):
                action_buffer = _sample_action_chunk(model, obs_history, cfg, metadata, device)
                action_buffer = action_buffer[:action_exec_horizon]
                buffer_index = 0
            action = np.asarray(action_buffer[buffer_index], dtype=np.float32)
            action = _clip_action(action, env.action_space)
            buffer_index += 1
            obs, reward, terminated, truncated, info = env.step(_to_env_action(action, env))
            obs_history.append(_flatten_obs(obs))
            obs_history = obs_history[-observation_horizon:]
            total_reward += float(_scalar(reward))
            success = success or _success_from_step(terminated, info)
            if success or bool(_scalar(truncated)):
                if writer is not None:
                    writer.append_data(_render_frame(env))
                break
    finally:
        if writer is not None:
            writer.close()
        env.close()
    return {
        "rollout_id": f"env_{seed}",
        "video_path": str(output_path.relative_to(PROJECT_ROOT)) if output_path is not None else None,
        "return": total_reward,
        "success": bool(success),
        "steps": step + 1,
    }


@torch.no_grad()
def _sample_action_chunk(model, obs_history: list[np.ndarray], cfg: dict[str, Any], metadata: dict[str, Any], device: torch.device) -> list[np.ndarray]:
    obs_vec = np.concatenate([_normalize_zscore(obs, metadata["normalization"]["obs"]) for obs in obs_history], axis=0)
    obs_tensor = torch.from_numpy(obs_vec[None].astype(np.float32)).to(device)
    target = euler_sample(model, obs_tensor, target_shape=model.target_shape, num_steps=int(cfg["sampling"]["num_flow_steps"])).cpu().numpy()
    norm_actions = _decode_target(target, cfg, metadata, model=model)[0]
    if is_raw_anchored_residual(cfg):
        residual = torch.from_numpy(norm_actions[None].astype(np.float32)).to(device)
        norm_actions = combine_raw_base_and_residual(residual, obs_tensor, getattr(model, "raw_base_policy", None), cfg)[0].cpu().numpy()
    elif is_sparse_dct_anchored_residual(cfg):
        residual = torch.from_numpy(norm_actions[None].astype(np.float32)).to(device)
        norm_actions = combine_base_and_residual(residual, obs_tensor, getattr(model, "base_policy", None), cfg)[0].cpu().numpy()
    if cfg.get("sampling", {}).get("clip_normalized_actions", True):
        norm_actions = np.clip(norm_actions, -1.0, 1.0)
    return list(_unnormalize_minmax(norm_actions, metadata["normalization"]["action"]))


def _decode_target(target: np.ndarray, cfg: dict[str, Any], metadata: dict[str, Any], model=None) -> np.ndarray:
    return decode_action_target(
        target,
        target_type=cfg["target"]["type"],
        horizon=int(cfg["chunking"]["horizon"]),
        action_dim=int(metadata["dataset"]["action_dim"]),
        dct_k=cfg["target"].get("dct_k"),
    )


def _flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [_flatten_obs(obs[key]) for key in sorted(obs.keys())]
        return np.concatenate(parts, axis=0)
    if isinstance(obs, torch.Tensor):
        arr = obs.detach().cpu().numpy()
    else:
        arr = np.asarray(obs)
    arr = np.squeeze(arr)
    return arr.reshape(-1).astype(np.float64)


def _normalize_zscore(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    return (x - np.asarray(stats["mean"])) / np.maximum(np.asarray(stats["std"]), float(stats["eps"]))


def _unnormalize_minmax(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    minimum = np.asarray(stats["min"])
    maximum = np.asarray(stats["max"])
    scale = maximum - minimum
    safe_scale = np.where(scale < float(stats["eps"]), 1.0, scale)
    return (x + 1.0) * 0.5 * safe_scale + minimum


def _clip_action(action: np.ndarray, action_space) -> np.ndarray:
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    return np.clip(action.reshape(-1), low, high).astype(np.float32)


def _to_env_action(action: np.ndarray, env):
    env_device = getattr(env.unwrapped, "device", None)
    if env_device is not None:
        return torch.as_tensor(action, dtype=torch.float32, device=env_device).reshape(1, -1)
    return action


def _success_from_step(terminated: Any, info: dict[str, Any]) -> bool:
    if bool(_scalar(terminated)):
        return True
    if isinstance(info, dict) and "success" in info:
        return bool(_scalar(info["success"]))
    return False


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def _render_frame(env) -> np.ndarray:
    frame = env.render()
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    return frame.astype(np.uint8)


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray([float(row["return"]) for row in results], dtype=np.float64)
    successes = np.asarray([bool(row["success"]) for row in results], dtype=np.float64)
    steps = np.asarray([int(row["steps"]) for row in results], dtype=np.float64)
    return {
        "num_rollouts": int(len(results)),
        "success_rate": float(np.mean(successes)) if len(results) else 0.0,
        "num_successes": int(np.sum(successes)) if len(results) else 0,
        "mean_return": float(np.mean(returns)) if len(results) else 0.0,
        "std_return": float(np.std(returns)) if len(results) else 0.0,
        "mean_steps": float(np.mean(steps)) if len(results) else 0.0,
        "std_steps": float(np.std(steps)) if len(results) else 0.0,
        "min_steps": int(np.min(steps)) if len(results) else 0,
        "max_steps": int(np.max(steps)) if len(results) else 0,
    }


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_output_path(ckpt_path: Path, metadata: dict[str, Any]) -> Path:
    run_name = ckpt_path.parent.name
    task = str(metadata["dataset"].get("env_id", "maniskill")).lower()
    return PROJECT_ROOT / "outputs" / "eval" / task / run_name / "simulation_rollout.mp4"


if __name__ == "__main__":
    main()
