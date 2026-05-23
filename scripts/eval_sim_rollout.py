#!/usr/bin/env python
"""Save headless robosuite simulation rollout videos for a trained FM policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from scipy.fft import idct

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fgac.models.flow_matching import FlowMatchingMLP, TemporalUNetFlow, euler_sample
from fgac.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to train_flow_matching best.pt")
    parser.add_argument("--output", default=None, help="Output mp4 path. Defaults under outputs/videos.")
    parser.add_argument(
        "--reset-mode",
        choices=["dataset", "env"],
        default=None,
        help="dataset resets to demo initial simulator states; env uses standard random env.reset().",
    )
    parser.add_argument("--demo", default=None, help="Dataset demo key to reset from, e.g. demo_0")
    parser.add_argument("--horizon", type=int, default=None, help="Rollout horizon override")
    parser.add_argument("--n-rollouts", type=int, default=None, help="Number of rollout videos")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of rollout videos to save.")
    parser.add_argument("--camera-names", nargs="+", default=None, help="Camera names for video")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = PROJECT_ROOT / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    metadata = ckpt["metadata"]
    sim_cfg = cfg.get("simulation_video", {})

    if metadata["action"]["source"] != "legacy":
        raise ValueError(
            "Simulation rollout currently requires executable legacy env actions. "
            f"Checkpoint action source is {metadata['action']['source']}."
        )

    device_name = args.device or cfg["training"].get("device", "cuda")
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = _load_model(ckpt, metadata, cfg, device)
    dataset_path = PROJECT_ROOT / cfg["dataset"]["path"]

    horizon = int(args.horizon or sim_cfg.get("horizon", 120))
    n_rollouts = int(args.n_rollouts or sim_cfg.get("n_rollouts", 1))
    max_videos = int(args.max_videos if args.max_videos is not None else sim_cfg.get("max_videos", n_rollouts))
    reset_mode = args.reset_mode or sim_cfg.get("reset_mode", "dataset")
    camera_names = args.camera_names or sim_cfg.get("camera_names", ["agentview"])
    height = int(args.height or sim_cfg.get("height", 512))
    width = int(args.width or sim_cfg.get("width", 512))
    fps = int(sim_cfg.get("fps", 20))
    video_skip = int(sim_cfg.get("video_skip", 1))
    action_exec_horizon = int(sim_cfg.get("action_exec_horizon", 4))

    output_base = Path(args.output) if args.output else _default_output_path(ckpt_path)
    output_base = PROJECT_ROOT / output_base if not output_base.is_absolute() else output_base
    output_base.parent.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed if args.seed is not None else sim_cfg.get("seed", cfg["run"].get("seed", 0)))
    np.random.seed(seed)
    torch.manual_seed(seed)

    rollout_ids = _choose_rollout_ids(dataset_path, args.demo, n_rollouts, reset_mode)
    results = []
    for i, rollout_id in enumerate(rollout_ids):
        output_path = output_base if i < max_videos else None
        if output_path is not None and n_rollouts > 1:
            output_path = output_base.with_name(f"{output_base.stem}_{rollout_id}{output_base.suffix}")
        result = _rollout_one(
            model=model,
            cfg=cfg,
            metadata=metadata,
            dataset_path=dataset_path,
            rollout_id=rollout_id,
            reset_mode=reset_mode,
            output_path=output_path,
            horizon=horizon,
            action_exec_horizon=action_exec_horizon,
            camera_names=camera_names,
            height=height,
            width=width,
            fps=fps,
            video_skip=video_skip,
            device=device,
            seed=seed + i,
        )
        results.append(result)
        if output_path is not None:
            print(f"Saved simulation video: {output_path.relative_to(PROJECT_ROOT)}")
        else:
            print("Skipped simulation video save for this rollout")
        print(f"  rollout={rollout_id} return={result['return']:.4f} success={result['success']}")

    metrics_path = output_base.with_suffix(".json")
    payload = {
        "checkpoint": str(ckpt_path),
        "reset_mode": reset_mode,
        "horizon": horizon,
        "action_exec_horizon": action_exec_horizon,
        "max_videos": max_videos,
        "seed": seed,
        "summary": _summarize_rollouts(results),
        "rollouts": results,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Rollout metrics: {metrics_path.relative_to(PROJECT_ROOT)}")


def _load_model(ckpt: dict[str, Any], metadata: dict[str, Any], cfg: dict[str, Any], device: torch.device):
    action_dim = int(metadata["dataset"]["action_dim"])
    horizon = int(cfg["chunking"]["horizon"])
    target_type = cfg["target"]["type"]
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "mlp")
    if target_type == "raw":
        target_dim = horizon * action_dim
        target_seq_len = horizon
    elif target_type == "dct_lowfreq":
        target_seq_len = int(cfg["target"]["dct_k"])
        target_dim = target_seq_len * action_dim
    else:
        raise ValueError(f"Unsupported target type: {target_type}")
    if model_type == "mlp":
        model = FlowMatchingMLP(
            obs_dim=int(metadata["dataset"]["obs_dim"]),
            target_dim=target_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            time_embed_dim=int(model_cfg.get("time_embed_dim", 64)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)
    elif model_type == "temporal_unet":
        model = TemporalUNetFlow(
            obs_dim=int(metadata["dataset"]["obs_dim"]),
            action_dim=action_dim,
            base_dim=int(model_cfg.get("base_dim", 128)),
            dim_mults=tuple(int(v) for v in model_cfg.get("dim_mults", [1, 2, 4])),
            time_embed_dim=int(model_cfg.get("time_embed_dim", 128)),
            cond_dim=int(model_cfg.get("cond_dim", 256)),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            groups=int(model_cfg.get("groups", 8)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)
        model.target_shape = (target_seq_len, action_dim)
    else:
        raise ValueError(f"Unsupported model.type: {model_type}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _summarize_rollouts(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "num_rollouts": 0,
            "success_rate": 0.0,
            "mean_return": 0.0,
            "mean_steps": 0.0,
        }
    returns = np.asarray([float(row["return"]) for row in results], dtype=np.float64)
    successes = np.asarray([bool(row["success"]) for row in results], dtype=np.float64)
    steps = np.asarray([int(row["steps"]) for row in results], dtype=np.float64)
    return {
        "num_rollouts": int(len(results)),
        "success_rate": float(np.mean(successes)),
        "num_successes": int(np.sum(successes)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "min_steps": int(np.min(steps)),
        "max_steps": int(np.max(steps)),
    }


def _rollout_one(
    model,
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    dataset_path: Path,
    rollout_id: str,
    reset_mode: str,
    output_path: Path | None,
    horizon: int,
    action_exec_horizon: int,
    camera_names: list[str],
    height: int,
    width: int,
    fps: int,
    video_skip: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    dummy_spec = {"obs": {"low_dim": cfg["dataset"]["obs_keys"], "rgb": []}}
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)

    try:
        if hasattr(env, "seed"):
            env.seed(seed)
        if reset_mode == "dataset":
            initial_state = _initial_state_from_demo(dataset_path, rollout_id)
            obs = env.reset_to(initial_state)
        elif reset_mode == "env":
            obs = env.reset()
        else:
            raise ValueError(f"Unsupported reset_mode: {reset_mode}")
        observation_horizon = int(metadata["dataset"].get("observation_horizon", cfg.get("chunking", {}).get("observation_horizon", 1)))
        obs_history = [obs for _ in range(observation_horizon)]
        total_reward = 0.0
        success = False
        action_buffer: list[np.ndarray] = []
        buffer_index = 0
        writer = imageio.get_writer(output_path, fps=fps, macro_block_size=8) if output_path is not None else None
        try:
            for step in range(horizon):
                if step % max(video_skip, 1) == 0:
                    if writer is not None:
                        writer.append_data(_render_frame(env, camera_names, height, width))
                if buffer_index >= len(action_buffer):
                    action_buffer = _sample_action_chunk(
                        model=model,
                        obs_history=obs_history,
                        cfg=cfg,
                        metadata=metadata,
                        device=device,
                    )
                    action_buffer = action_buffer[:action_exec_horizon]
                    buffer_index = 0
                action = action_buffer[buffer_index]
                buffer_index += 1
                obs, reward, done, info = env.step(action)
                obs_history.append(obs)
                obs_history = obs_history[-observation_horizon:]
                total_reward += float(reward)
                success = bool(info.get("is_success", {}).get("task", False))
                if success:
                    if writer is not None:
                        writer.append_data(_render_frame(env, camera_names, height, width))
                    break
        finally:
            if writer is not None:
                writer.close()
        return {
            "rollout_id": rollout_id,
            "reset_mode": reset_mode,
            "video_path": str(output_path.relative_to(PROJECT_ROOT)) if output_path is not None else None,
            "return": total_reward,
            "success": success,
            "steps": step + 1,
        }
    finally:
        env.env.close()


@torch.no_grad()
def _sample_action_chunk(
    model,
    obs_history: list[dict[str, np.ndarray]],
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    device,
):
    obs_vec = _obs_history_to_vector(obs_history, cfg["dataset"]["obs_keys"], metadata["normalization"]["obs"])
    obs_tensor = torch.from_numpy(obs_vec[None].astype(np.float32)).to(device)
    if cfg.get("model", {}).get("type", "mlp") == "temporal_unet":
        target = euler_sample(
            model,
            obs_tensor,
            target_shape=model.target_shape,
            num_steps=int(cfg["sampling"]["num_flow_steps"]),
        ).cpu().numpy()
    else:
        target = euler_sample(
            model,
            obs_tensor,
            target_dim=model.net[-1].out_features,
            num_steps=int(cfg["sampling"]["num_flow_steps"]),
        ).cpu().numpy()
    norm_actions = _decode_target(target, cfg, metadata)[0]
    if cfg.get("sampling", {}).get("clip_normalized_actions", True):
        norm_actions = np.clip(norm_actions, -1.0, 1.0)
    return list(_unnormalize_minmax(norm_actions, metadata["normalization"]["action"]))


def _decode_target(target: np.ndarray, cfg: dict[str, Any], metadata: dict[str, Any]) -> np.ndarray:
    horizon = int(cfg["chunking"]["horizon"])
    action_dim = int(metadata["dataset"]["action_dim"])
    if cfg["target"]["type"] == "raw":
        if target.ndim == 3:
            return target
        return target.reshape(target.shape[0], horizon, action_dim)
    if cfg["target"]["type"] == "dct_lowfreq":
        k = int(cfg["target"]["dct_k"])
        coeffs = np.zeros((target.shape[0], horizon, action_dim), dtype=np.float32)
        coeffs[:, :k, :] = target if target.ndim == 3 else target.reshape(target.shape[0], k, action_dim)
        return idct(coeffs, axis=1, norm="ortho")
    raise ValueError(f"Unsupported target type: {cfg['target']['type']}")


def _obs_history_to_vector(
    obs_history: list[dict[str, np.ndarray]],
    obs_keys: list[str],
    stats: dict[str, Any],
) -> np.ndarray:
    frames = []
    for obs in obs_history:
        frames.append(_normalize_zscore(_obs_to_vector(obs, obs_keys), stats))
    return np.concatenate(frames, axis=0)


def _obs_to_vector(obs: dict[str, np.ndarray], obs_keys: list[str]) -> np.ndarray:
    parts = []
    for key in obs_keys:
        if key not in obs:
            raise KeyError(f"Rollout observation missing key '{key}'. Available keys: {list(obs.keys())}")
        parts.append(np.asarray(obs[key]).reshape(-1))
    return np.concatenate(parts, axis=0)


def _normalize_zscore(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    return (x - np.asarray(stats["mean"])) / np.maximum(np.asarray(stats["std"]), float(stats["eps"]))


def _unnormalize_minmax(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    minimum = np.asarray(stats["min"])
    maximum = np.asarray(stats["max"])
    scale = maximum - minimum
    safe_scale = np.where(scale < float(stats["eps"]), 1.0, scale)
    y = (x + 1.0) * 0.5 * safe_scale + minimum
    constant_dims = scale < float(stats["eps"])
    if np.any(constant_dims):
        y[..., constant_dims] = minimum[constant_dims]
    return y


def _render_frame(env, camera_names: list[str], height: int, width: int) -> np.ndarray:
    frames = [
        env.render(mode="rgb_array", height=height, width=width, camera_name=camera_name)
        for camera_name in camera_names
    ]
    return np.concatenate(frames, axis=1)


def _initial_state_from_demo(dataset_path: Path, demo: str) -> dict[str, Any]:
    with h5py.File(dataset_path, "r") as f:
        group = f[f"data/{demo}"]
        state = {"states": group["states"][0]}
        state["model"] = group.attrs["model_file"]
        state["ep_meta"] = group.attrs.get("ep_meta", None)
    return state


def _choose_rollout_ids(dataset_path: Path, demo: str | None, n_rollouts: int, reset_mode: str) -> list[str]:
    if reset_mode == "env":
        if demo is not None:
            raise ValueError("--demo is only valid with --reset-mode dataset")
        return [f"env_{idx:03d}" for idx in range(n_rollouts)]
    if reset_mode != "dataset":
        raise ValueError(f"Unsupported reset_mode: {reset_mode}")
    with h5py.File(dataset_path, "r") as f:
        demos = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[-1]))
    if demo is not None:
        return [demo]
    return demos[:n_rollouts]


def _default_output_path(ckpt_path: Path) -> Path:
    run_name = ckpt_path.parent.name
    return Path("outputs/eval/simulation") / run_name / "simulation_rollout.mp4"


if __name__ == "__main__":
    main()
