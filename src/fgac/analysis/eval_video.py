"""Headless qualitative evaluation videos for action chunk predictions."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_action_chunk_eval_video(
    true_actions: np.ndarray,
    pred_actions: np.ndarray,
    output_path: str | Path,
    dim_names: list[str],
    num_examples: int = 4,
    fps: int = 4,
) -> None:
    """Save a headless mp4 comparing predicted and true validation chunks.

    This is not an environment rollout video. It is a qualitative validation
    video for Experiment C/D that shows whether generated chunks match the
    dataset action sequence and whether predictions are overly jittery.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num = min(num_examples, true_actions.shape[0], pred_actions.shape[0])
    if num <= 0:
        raise ValueError("No examples available for eval video.")

    true_actions = true_actions[:num]
    pred_actions = pred_actions[:num]
    horizon = true_actions.shape[1]
    action_dim = true_actions.shape[2]
    dims_to_plot = list(range(min(action_dim, 7)))
    labels = [dim_names[i] if i < len(dim_names) else f"action_{i}" for i in dims_to_plot]

    with imageio.get_writer(output_path, fps=fps, macro_block_size=8) as writer:
        for t in range(horizon):
            fig, axes = plt.subplots(num, 1, figsize=(10, max(2.4 * num, 3.0)), squeeze=False)
            for row in range(num):
                ax = axes[row, 0]
                xs = np.arange(horizon)
                for dim, label in zip(dims_to_plot, labels):
                    ax.plot(xs, true_actions[row, :, dim], linestyle="-", linewidth=1.3, alpha=0.8, label=f"gt {label}")
                    ax.plot(xs, pred_actions[row, :, dim], linestyle="--", linewidth=1.1, alpha=0.75, label=f"pred {label}")
                ax.axvline(t, color="black", linewidth=1.5, alpha=0.75)
                err = float(np.mean((true_actions[row, : t + 1] - pred_actions[row, : t + 1]) ** 2))
                ax.set_title(f"val chunk {row} | timestep {t + 1}/{horizon} | prefix MSE {err:.4f}")
                ax.set_xlabel("chunk timestep")
                ax.set_ylabel("normalized action")
                ax.grid(True, alpha=0.25)
                if row == 0:
                    ax.legend(loc="upper right", ncol=2, fontsize=7)
            fig.tight_layout()
            writer.append_data(_figure_to_rgb(fig))
            plt.close(fig)


def _figure_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.asarray(rgba[..., :3])
