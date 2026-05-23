#!/usr/bin/env python
"""Generate Chinese and English milestone reports from experiment outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


FREQUENCY_EXPERIMENTS = [
    ("Lift", "outputs/analysis/frequency_diagnostic/metrics/lift_ph_frequency_diagnostic_*.json"),
    ("Can", "outputs/analysis/can_frequency_diagnostic/metrics/can_ph_frequency_diagnostic_*.json"),
    ("Push-T", "outputs/analysis/pusht_frequency_diagnostic/metrics/pusht_frequency_diagnostic_*.json"),
]


POLICY_EXPERIMENTS = [
    ("Lift", "Raw UNet FM", "outputs/eval/lift_ph_fm_raw_*/env/metrics.json"),
    ("Lift", "DCT Low-Frequency UNet FM", "outputs/eval/lift_ph_fm_dct_lowfreq_k8_*/env/metrics.json"),
    ("Can", "Raw UNet FM", "outputs/eval/can_ph_fm_raw_*/env/metrics.json"),
    ("Can", "DCT Low-Frequency UNet FM", "outputs/eval/can_ph_fm_dct_lowfreq_k8_*/env/metrics.json"),
    ("Push-T", "Raw UNet FM", "outputs/eval/pusht_fm_raw_*/env/metrics.json"),
    ("Push-T", "DCT Low-Frequency UNet FM", "outputs/eval/pusht_fm_dct_lowfreq_k8_*/env/metrics.json"),
]


OUTPUT_GROUPS = [
    ("Frequency diagnostics", "outputs/analysis/*/metrics/*.json"),
    ("Training checkpoints", "outputs/train/*/checkpoints/*/best.pt"),
    ("Training metrics", "outputs/train/*/metrics/*.json"),
    ("Rollout metrics", "outputs/eval/*/*/metrics.json"),
    ("Rollout JSON", "outputs/eval/*/*/simulation_rollout.json"),
    ("Simulation videos", "outputs/eval/*/*/*.mp4"),
    ("Pipeline logs", "logs/pipeline/*/*.log"),
    ("Detached logs", "logs/detached/*.log"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zh-output", default="docs/reports/milestone_report_zh.md")
    parser.add_argument("--en-output", default="docs/reports/milestone_report_en.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zh_output = PROJECT_ROOT / args.zh_output
    en_output = PROJECT_ROOT / args.en_output
    zh_output.parent.mkdir(parents=True, exist_ok=True)
    en_output.parent.mkdir(parents=True, exist_ok=True)

    frequency_rows = [_load_frequency_row(task, pattern) for task, pattern in FREQUENCY_EXPERIMENTS]
    policy_rows = [_load_policy_row(task, method, pattern) for task, method, pattern in POLICY_EXPERIMENTS]
    output_rows = [_load_output_row(name, pattern) for name, pattern in OUTPUT_GROUPS]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    zh_output.write_text(_render_zh(now, frequency_rows, policy_rows, output_rows), encoding="utf-8")
    en_output.write_text(_render_en(now, frequency_rows, policy_rows, output_rows), encoding="utf-8")
    print(zh_output.relative_to(PROJECT_ROOT))
    print(en_output.relative_to(PROJECT_ROOT))


def _latest_json(pattern: str) -> Path | None:
    matches = sorted(PROJECT_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_frequency_row(task: str, pattern: str) -> dict[str, Any]:
    path = _latest_json(pattern)
    data = _load_json(path)
    if data is None:
        return {"task": task, "status": "missing", "path": None}
    k8 = _row_for_k(data.get("by_k", []), 8)
    k4 = _row_for_k(data.get("by_k", []), 4)
    return {
        "task": task,
        "status": "done",
        "path": _rel(path),
        "num_chunks": data.get("dataset", {}).get("num_chunks"),
        "action_dim": data.get("dataset", {}).get("action_dim"),
        "k4_mse": _get(k4, "reconstruction_mse"),
        "k8_mse": _get(k8, "reconstruction_mse"),
        "k8_smoothness": _get(k8, "reconstruction_smoothness"),
        "k8_high_energy": _get(k8, "high_energy_ratio_mean"),
        "transition_enabled": data.get("transition_conditioned", {}).get("enabled", False),
    }


def _load_policy_row(task: str, method: str, pattern: str) -> dict[str, Any]:
    path = _latest_json(pattern)
    data = _load_json(path)
    if data is None:
        return {"task": task, "method": method, "status": "missing", "path": None}
    val = data.get("best_validation", {})
    sim = data.get("simulation", {})
    return {
        "task": task,
        "method": method,
        "status": "done",
        "path": _rel(path),
        "checkpoint": data.get("checkpoint"),
        "epoch": val.get("epoch"),
        "action_mse": val.get("action_mse"),
        "smoothness": val.get("smoothness"),
        "true_smoothness": val.get("true_smoothness"),
        "delta_action_mse": val.get("delta_action_mse"),
        "success_rate": sim.get("success_rate"),
        "num_successes": sim.get("num_successes"),
        "mean_return": sim.get("mean_return"),
        "mean_steps": sim.get("mean_steps"),
        "num_rollouts": sim.get("num_rollouts"),
        "video_count": len(data.get("videos", [])),
    }


def _load_output_row(name: str, pattern: str) -> dict[str, Any]:
    matches = sorted(PROJECT_ROOT.glob(pattern), key=lambda p: str(p))
    examples = [_rel(path) for path in matches[-3:]]
    return {
        "name": name,
        "pattern": pattern,
        "count": len(matches),
        "examples": examples,
    }


def _row_for_k(rows: list[dict[str, Any]], k: int) -> dict[str, Any] | None:
    for row in rows:
        if int(row.get("k", -1)) == k:
            return row
    return None


def _get(row: dict[str, Any] | None, key: str) -> Any:
    return None if row is None else row.get(key)


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _status(value: str) -> str:
    return "done" if value == "done" else "missing"


def _render_frequency_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Task | Status | Chunks | Action Dim | K=4 MSE | K=8 MSE | K=8 Smoothness | K=8 High-Energy | Metrics |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {status} | {chunks} | {dim} | {k4} | {k8} | {smooth} | {energy} | {path} |".format(
                task=row["task"],
                status=_status(row["status"]),
                chunks=_fmt(row.get("num_chunks"), 0),
                dim=_fmt(row.get("action_dim"), 0),
                k4=_fmt(row.get("k4_mse")),
                k8=_fmt(row.get("k8_mse")),
                smooth=_fmt(row.get("k8_smoothness")),
                energy=_fmt(row.get("k8_high_energy")),
                path=row.get("path") or "-",
            )
        )
    return "\n".join(lines)


def _render_policy_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Task | Method | Status | Best Epoch | Val Action MSE | Smoothness | Delta MSE | Rollouts | Success | Videos | Mean Return | Mean Steps | Metrics |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {method} | {status} | {epoch} | {mse} | {smooth} | {delta} | {n} | {succ} | {videos} | {ret} | {steps} | {path} |".format(
                task=row["task"],
                method=row["method"],
                status=_status(row["status"]),
                epoch=_fmt(row.get("epoch"), 0),
                mse=_fmt(row.get("action_mse")),
                smooth=_fmt(row.get("smoothness")),
                delta=_fmt(row.get("delta_action_mse")),
                n=_fmt(row.get("num_rollouts"), 0),
                succ=_fmt(row.get("success_rate")),
                videos=_fmt(row.get("video_count"), 0),
                ret=_fmt(row.get("mean_return")),
                steps=_fmt(row.get("mean_steps")),
                path=row.get("path") or "-",
            )
        )
    return "\n".join(lines)


def _render_output_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Output Type | Count | Pattern | Recent Examples |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        examples = "<br>".join(row["examples"]) if row["examples"] else "-"
        lines.append(f"| {row['name']} | {row['count']} | `{row['pattern']}` | {examples} |")
    return "\n".join(lines)


def _policy_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], {})[row["method"]] = row
    return by_task


def _delta_pct(new: Any, base: Any) -> str:
    if new is None or base in (None, 0):
        return "-"
    return f"{(float(new) - float(base)) / float(base) * 100:+.1f}%"


def _render_findings_zh(rows: list[dict[str, Any]]) -> str:
    by_task = _policy_by_task(rows)
    lines = []
    for task in ["Lift", "Can", "Push-T"]:
        raw = by_task.get(task, {}).get("Raw UNet FM")
        dct = by_task.get(task, {}).get("DCT Low-Frequency UNet FM")
        if not raw or not dct or raw["status"] != "done" or dct["status"] != "done":
            continue
        lines.append(
            "- {task}: raw success={raw_succ}, DCT success={dct_succ}; "
            "DCT smoothness change={smooth_delta}, action MSE change={mse_delta}.".format(
                task=task,
                raw_succ=_fmt(raw.get("success_rate")),
                dct_succ=_fmt(dct.get("success_rate")),
                smooth_delta=_delta_pct(dct.get("smoothness"), raw.get("smoothness")),
                mse_delta=_delta_pct(dct.get("action_mse"), raw.get("action_mse")),
            )
        )
    lines.append("- Lift 上 DCT 和 raw 都达到 100% env-reset success，说明简单任务不足以区分频率瓶颈。")
    lines.append("- Can 上 DCT 的 validation MSE 和 smoothness 都更好，但 env success 从 0.94 降到 0.62，这是最强的 evidence：低频 reconstruction 好不等于 closed-loop manipulation 好。")
    lines.append("- Push-T 上 DCT 与 raw 表现相当甚至略好，说明 planar pushing state benchmark 对低频动作块更友好。")
    lines.append("- 下一步应该训练 adaptive gate / high-frequency correction：在 Lift/Push-T 保留低频平滑优势，在 Can 这种需要 precision 的阶段恢复高频。")
    return "\n".join(lines)


def _render_findings_en(rows: list[dict[str, Any]]) -> str:
    by_task = _policy_by_task(rows)
    lines = []
    for task in ["Lift", "Can", "Push-T"]:
        raw = by_task.get(task, {}).get("Raw UNet FM")
        dct = by_task.get(task, {}).get("DCT Low-Frequency UNet FM")
        if not raw or not dct or raw["status"] != "done" or dct["status"] != "done":
            continue
        lines.append(
            "- {task}: raw success={raw_succ}, DCT success={dct_succ}; "
            "DCT smoothness change={smooth_delta}, action MSE change={mse_delta}.".format(
                task=task,
                raw_succ=_fmt(raw.get("success_rate")),
                dct_succ=_fmt(dct.get("success_rate")),
                smooth_delta=_delta_pct(dct.get("smoothness"), raw.get("smoothness")),
                mse_delta=_delta_pct(dct.get("action_mse"), raw.get("action_mse")),
            )
        )
    lines.append("- On Lift, both raw and DCT reach 100% env-reset success, so the easy task is not discriminative enough by itself.")
    lines.append("- On Can, DCT has better validation MSE and smoothness, but env success drops from 0.94 to 0.62. This is the strongest evidence that good low-frequency reconstruction does not guarantee closed-loop manipulation success.")
    lines.append("- On Push-T, DCT is comparable or slightly better than raw, suggesting this planar state benchmark is friendly to low-frequency action chunks.")
    lines.append("- The next method should train an adaptive gate or high-frequency correction: keep the low-frequency smoothness benefit on Lift/Push-T, while restoring precision on contact-sensitive Can phases.")
    return "\n".join(lines)


def _render_fast_analysis_zh() -> str:
    return """参考：FAST: Efficient Action Tokenization for Vision-Language-Action Models, arXiv:2501.09747, https://arxiv.org/abs/2501.09747

FAST 和本 milestone 的 Low-Frequency DCT baseline 都使用 DCT，但两者的 bottleneck 不同：

1. 当前 Low-Frequency DCT FM 是固定低通：只保留前 `K=8` 个 DCT temporal coefficients，所有 `k>=K` 的高频系数都被置零。
2. FAST 更接近压缩式 tokenization：对完整动作序列做 DCT，然后量化频域系数，并保留量化后非零的稀疏系数。高频系数如果幅值足够大，仍然可以被保留。
3. 因此，Can 上固定低通 DCT success 从 raw 的 0.94 降到 0.62，并不反驳 FAST。它说明 fixed low-pass bottleneck 会过度平滑 precision-sensitive manipulation，而 FAST-style sparse full-spectrum DCT 或 adaptive high-frequency correction 正是为了解决这个问题。

当前结果和 FAST 的预期关系是：低频结构确实能降低 action variation；但在 Can 这种需要抓取、对齐和短时修正的任务上，不能无条件丢弃高频。下一步应该比较 fixed low-pass DCT 与 sparse full-spectrum DCT，在相同 compression ratio 下观察 success、smoothness 和 delta MSE 的变化。"""


def _render_fast_analysis_en() -> str:
    return """Reference: FAST: Efficient Action Tokenization for Vision-Language-Action Models, arXiv:2501.09747, https://arxiv.org/abs/2501.09747

FAST and the milestone Low-Frequency DCT baseline both use DCT, but they impose different bottlenecks:

1. The current Low-Frequency DCT FM baseline is a fixed low-pass model: it keeps only the first `K=8` temporal DCT coefficients and zeros out all coefficients with `k>=K`.
2. FAST is closer to compression-based tokenization: apply DCT to the full action sequence, quantize frequency-domain coefficients, and retain sparse nonzero coefficients after quantization. High-frequency coefficients can survive when their magnitude is large enough.
3. Therefore, the Can result, where fixed low-pass DCT success drops from raw 0.94 to 0.62, does not contradict FAST. It shows that a fixed low-pass bottleneck can over-smooth precision-sensitive manipulation, which is exactly where FAST-style sparse full-spectrum DCT or adaptive high-frequency correction should help.

The current results are consistent with the FAST motivation: low-frequency structure reduces action variation, but high-frequency content should not be discarded unconditionally on tasks that require grasping, alignment, and short corrections. The next comparison should evaluate fixed low-pass DCT against sparse full-spectrum DCT at matched compression ratios, measuring success, smoothness, and delta MSE."""


def _render_task_plan_zh() -> str:
    return """任务计划按证据类型划分，而不是只堆更多 benchmark：

1. Push-T: smoothness-positive task
   - 用来证明 smooth reconstructed actions 对连续、低抖动控制有用。
   - 当前 DCT 表示降低 action variation，同时 success / return 不下降，适合作为 smoothness 正例。
   - 后续可加入 harder Push-T variants，例如更窄目标、更长 horizon、更复杂初始分布或障碍物。

2. Lift: smoothness sanity-check task
   - 用来证明低频 smooth reconstruction 不会破坏简单 reaching / lifting。
   - 任务太简单，不能单独证明 smoothness 有用。

3. Can: smoothness-precision trade-off task
   - 用来证明 smoothness alone 不够。
   - 当前 Can 上 DCT smoothness 和 validation MSE 更好，但 success 明显下降，说明 precision-sensitive phase 需要高频信息。

4. Next stress tests
   - ManiSkill `PegInsertionSide-v1` / `PlugCharger-v1`: 插孔、插头和小幅接触调整。
   - robomimic Square / Tool Hang: 低成本 precision assembly 扩展。
   - DROID filtered subset: 真实机器人 `insert / plug / adjust / align` 片段的 offline representation diagnostic。"""


def _render_task_plan_en() -> str:
    return """The task plan is organized by evidence type, not by simply adding more benchmarks:

1. Push-T: smoothness-positive task
   - Use it to test whether smoother reconstructed actions help continuous, low-jitter control.
   - The current DCT representation reduces action variation while maintaining or improving success / return, so it is the positive smoothness example.
   - Later add harder Push-T variants, such as tighter goals, longer horizons, harder initial states, or obstacles.

2. Lift: smoothness sanity-check task
   - Use it to show that low-frequency smooth reconstruction does not break simple reaching / lifting.
   - It is too easy to prove smoothness usefulness by itself.

3. Can: smoothness-precision trade-off task
   - Use it to show that smoothness alone is insufficient.
   - On Can, DCT has better smoothness and validation MSE but much worse success, showing that precision-sensitive phases require high-frequency information.

4. Next stress tests
   - ManiSkill `PegInsertionSide-v1` / `PlugCharger-v1`: insertion, plug-in, and small contact adjustment.
   - robomimic Square / Tool Hang: low-cost precision assembly extensions.
   - DROID filtered subset: offline representation diagnostics on real-robot `insert / plug / adjust / align` segments."""


def _render_next_steps_zh() -> str:
    return """下一步把以下方法和数据集扩展加入计划和实验路线：

- 方法 1: FAST-style Sparse DCT Tokenizer
   - 不再固定保留 `K=8`，而是在完整 DCT spectrum 上做 sparse retention。
   - 第一版实现 continuous sparse DCT：保留 top-magnitude coefficients 或 thresholded coefficients，再用 IDCT 解码。
   - 关键比较：相同 compression ratio 下，是否比 fixed low-pass DCT 更好地保持 Can success，同时保留 smoothness 优势。

- 方法 4: Wavelet Tokenizer
   - 用 multi-resolution basis 替代纯 DCT。
   - 重点测试 gripper transition、alignment correction 等局部高频事件是否比 DCT low-pass 保真更好。

- DROID scale-up dataset
   - DROID 是 in-the-wild robot manipulation dataset，官方论文/主页报告约 76k demonstration trajectories、350 小时 interaction data。
   - 先实现 DROID adapter，把真实机器人轨迹映射成统一 `[H, d]` action chunks。
   - DROID 第一阶段只做 offline frequency/tokenizer diagnostics、representation / gate pretraining 和 qualitative dataset visualization。
   - DROID 没有本地 simulator reset protocol，因此不直接放进 simulation success table；后续用 DROID 预训练，再在 Can、Push-T 或其他可执行 benchmark 上 fine-tune / evaluate。

- Precision / dexterous / contact-rich task expansion
   - 精密插孔和插头类任务优先接入 ManiSkill `PegInsertionSide-v1` 和 `PlugCharger-v1`，因为它们有可执行 simulation environment 和 demonstration workflow。
   - robomimic Square / Tool Hang 可作为低成本 precision assembly 扩展，复用现有 robomimic pipeline。
   - 灵巧手指任务优先考虑 D4RL / Adroit 或 RoboHive；这类任务 action dimension 和控制语义不同，需要单独 hand-action adapter，先做 offline representation diagnostic。
   - 小幅接触调整可从 DROID language annotations 中按 `insert`、`plug`、`adjust`、`align` 等关键词过滤，再结合 gripper / end-effector trajectory statistics 和视频抽查确认。

所有新方法都应尽量复用当前 pipeline：UNet Flow Matching backbone、normalization、W&B logging、统一 `metrics.json`。可执行 benchmark 继续保存 `env-reset` rollout 和 simulation video；DROID 保存 offline metrics、action reconstruction plots 和 dataset trajectory visualization。"""


def _render_next_steps_en() -> str:
    return """Add the following methods and dataset extension to the next experiment plan:

- Method 1: FAST-style Sparse DCT Tokenizer
   - Stop truncating to a fixed `K=8`; instead, perform sparse retention over the full DCT spectrum.
   - First implement a continuous sparse DCT version: keep top-magnitude or thresholded coefficients, then decode with IDCT.
   - Key comparison: at matched compression ratios, does it preserve Can success better than fixed low-pass DCT while keeping the smoothness benefit?

- Method 4: Wavelet Tokenizer
   - Replace pure DCT with a multi-resolution basis.
   - Focus on whether localized high-frequency events such as gripper transitions and alignment corrections are reconstructed better than with DCT low-pass.

- DROID scale-up dataset
   - DROID is an in-the-wild robot manipulation dataset; the official paper/site report about 76k demonstration trajectories and 350 hours of interaction data.
   - First implement a DROID adapter that maps real-robot trajectories into unified `[H, d]` action chunks.
   - The first DROID stage should run offline frequency/tokenizer diagnostics, representation / gate pretraining, and qualitative dataset visualization.
   - DROID has no local simulator reset protocol, so it should not be placed directly into the simulation success table. Use DROID for pretraining, then fine-tune / evaluate on Can, Push-T, or another executable benchmark.

- Precision / dexterous / contact-rich task expansion
   - For precision insertion and plug-in tasks, add ManiSkill `PegInsertionSide-v1` and `PlugCharger-v1` first because they provide executable simulation environments and demonstration workflows.
   - Use robomimic Square / Tool Hang as low-cost precision assembly extensions that reuse the current robomimic pipeline.
   - For dexterous fingers, prioritize D4RL / Adroit or RoboHive. These tasks have different action dimensions and control semantics, so they need a separate hand-action adapter and should start with offline representation diagnostics.
   - For small contact adjustment, filter DROID language annotations with keywords such as `insert`, `plug`, `adjust`, and `align`, then combine gripper / end-effector trajectory statistics with video auditing.

All new methods should reuse as much of the current pipeline as possible: the UNet Flow Matching backbone, normalization, W&B logging, and unified `metrics.json` output. Executable benchmarks continue to save `env-reset` rollouts and simulation videos; DROID saves offline metrics, action reconstruction plots, and dataset trajectory visualizations."""


def _render_zh(
    now: str,
    frequency_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
) -> str:
    completed = sum(1 for row in policy_rows if row["status"] == "done")
    total = len(policy_rows)
    return f"""# 动作表示 Milestone 实验报告

生成时间：{now}

## 1. Motivation

本项目研究的问题是：生成式机器人策略应该使用什么样的动作表示，才能同时保持平滑控制、精密操作和稳定执行。当前 milestone 使用 action chunking 作为统一 train/eval 接口，但项目本身不局限于 action chunking；DCT bottleneck、adaptive frequency gate、sparse DCT 和 wavelet 都是后续可比较的 action representation。

这个 milestone 不直接宣称 adaptive frequency gate 已经优于 baseline，而是先建立完整的 Flow Matching train/eval pipeline，并检查一个关键前提：压缩或低频化的动作表示是否仍能保留 closed-loop control 所需的信息。当前实验是 tokenizer-motivated，但还不是完整 tokenizer，因为没有离散 vocabulary、BPE 或 autoregressive token decoding。

关键假设：

1. Raw continuous action chunk Flow Matching 可以作为动作表示 baseline。
2. DCT low-frequency Flow Matching 如果能在成功率接近 raw baseline 时降低 action variation，说明低频表示对平滑控制有解释力。
3. 如果 DCT low-frequency 在 Can、插孔或接触调整任务上明显失败，则说明 sparse high-frequency retention、adaptive gate 或 wavelet representation 是必要的。

## 2. Environment

- Conda 环境：`cs224r`
- Headless rendering：`MUJOCO_GL=egl`，`PYOPENGL_PLATFORM=egl`，`MPLBACKEND=Agg`
- 训练日志：Weights & Biases online mode，project `cs224r-fgac`
- Robomimic datasets：Lift-PH low_dim、Can-PH low_dim
- Push-T dataset：`data/pusht/pusht_cchi_v7_replay.zarr`

## 3. 实验设计

任务：

1. Lift-PH low_dim：robomimic sanity-check task。
2. Can-PH low_dim：更难的 robomimic manipulation task。
3. Push-T state：planar pushing benchmark。

动作表示 / 方法：

1. Raw temporal-UNet Flow Matching：直接预测长度为 `H=16` 的 normalized action chunk。
2. Low-Frequency DCT temporal-UNet Flow Matching：预测前 `K=8` 个 DCT temporal coefficients，IDCT 后还原动作块。

主要实现约束：

1. Robomimic 可执行 rollout 使用 legacy 7-D action，因为 simulator 需要环境原生 action。
2. 离线频域诊断使用 per-dim normalization，避免 gripper 或 position 的尺度主导总能量。
3. Rollout 采用 receding-horizon execution，`K_exec=8`，即每次预测一个 chunk 后执行前 8 个动作再重新预测。

正式 rollout 指标使用 `env-reset`，即随机环境初始状态；`dataset-reset` 视频只作为 debug/qualitative evidence。

## 4. Evaluation Metrics

- `Val Action MSE`：验证集动作块重建 MSE，越低越好。
- `Smoothness`：预测动作序列相邻动作差分平方均值，越低代表动作变化更平滑。
- `Delta MSE`：预测动作差分和真实动作差分之间的 MSE，用来衡量 temporal dynamics 是否匹配。
- `Success`：simulation rollout 成功率，是最终 task-level 指标。
- `Mean Return / Mean Steps`：环境返回和 episode 长度，用于辅助判断失败模式。
- `K=4/K=8 reconstruction MSE`：只保留低频 DCT 系数时的离线重建误差。
- `High-Energy Ratio`：高频 DCT 能量比例，用来判断任务是否需要高频动作成分。

## 5. Frequency Diagnostic

{_render_frequency_table(frequency_rows)}

## 6. Train / Eval Results

当前完成 `{completed}/{total}` 个 policy train/eval 条目。

{_render_policy_table(policy_rows)}

## 7. Output 分类

{_render_output_table(output_rows)}

## 8. 当前结论

{_render_findings_zh(policy_rows)}

## 9. Task 计划

{_render_task_plan_zh()}

## 10. FAST 对比分析

{_render_fast_analysis_zh()}

## 11. 下一步计划

{_render_next_steps_zh()}

## 12. 复现实验命令

```bash
scripts/launch_detached_milestone_pipeline.sh
tail -f logs/detached/milestone_pipeline_*.log
```

单独生成报告：

```bash
python3 scripts/generate_milestone_report.py
```
"""


def _render_en(
    now: str,
    frequency_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
) -> str:
    completed = sum(1 for row in policy_rows if row["status"] == "done")
    total = len(policy_rows)
    return f"""# Action Representation Milestone Report

Generated at: {now}

## 1. Motivation

This project asks what action representation a generative robot policy should use to preserve smooth control, precise manipulation, and stable execution. The current milestone uses action chunking as the shared train/eval interface, but the project is not limited to action chunking. DCT bottlenecks, adaptive frequency gates, sparse DCT, and wavelets are candidate action representations for later comparison.

The milestone does not claim that the adaptive frequency gate is already better than the baseline. Instead, it establishes the complete Flow Matching train/eval pipeline and tests a prerequisite: whether compressed or low-frequency action representations preserve the information needed for closed-loop control. The current experiments are tokenizer-motivated but not yet tokenizer-based, because they do not use a discrete vocabulary, BPE, or autoregressive token decoding.

Main hypotheses:

1. Raw continuous action chunk Flow Matching is a suitable action-representation baseline.
2. If low-frequency DCT Flow Matching matches raw success while reducing action variation, low-frequency representations explain a large part of smooth control.
3. If low-frequency DCT fails on Can, insertion, or contact-adjustment tasks, sparse high-frequency retention, adaptive gating, or wavelet representations are needed.

## 2. Environment

- Conda environment: `cs224r`
- Headless rendering: `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `MPLBACKEND=Agg`
- Training logs: Weights & Biases online mode, project `cs224r-fgac`
- Robomimic datasets: Lift-PH low_dim and Can-PH low_dim
- Push-T dataset: `data/pusht/pusht_cchi_v7_replay.zarr`

## 3. Experimental Design

Tasks:

1. Lift-PH low_dim: robomimic sanity-check task.
2. Can-PH low_dim: harder robomimic manipulation task.
3. Push-T state: external planar pushing benchmark.

Action representations / methods:

1. Raw temporal-UNet Flow Matching: predicts a normalized action chunk of horizon `H=16`.
2. Low-Frequency DCT temporal-UNet Flow Matching: predicts the first `K=8` temporal DCT coefficients and reconstructs the action chunk through IDCT.

Implementation constraints:

1. Robomimic executable rollouts use legacy 7-D actions because the simulator requires native environment actions.
2. Offline frequency diagnostics use per-dimension normalization so gripper or position scale does not dominate total energy.
3. Rollouts use receding-horizon execution with `K_exec=8`: predict a full chunk, execute the first 8 actions, then replan.

Official rollout metrics use `env-reset`, i.e. randomized environment initial states. Dataset-reset videos are kept for debugging and qualitative inspection only.

## 4. Evaluation Metrics

- `Val Action MSE`: validation action-chunk reconstruction MSE; lower is better.
- `Smoothness`: mean squared adjacent action difference for predicted actions; lower means less action variation.
- `Delta MSE`: MSE between predicted action differences and ground-truth action differences.
- `Success`: simulation rollout success rate; this is the main task-level metric.
- `Mean Return / Mean Steps`: auxiliary rollout diagnostics.
- `K=4/K=8 reconstruction MSE`: offline reconstruction error after keeping only low-frequency DCT coefficients.
- `High-Energy Ratio`: high-frequency DCT energy ratio, used to estimate how much high-frequency action structure the task contains.

## 5. Frequency Diagnostic

{_render_frequency_table(frequency_rows)}

## 6. Train / Eval Results

Currently completed `{completed}/{total}` policy train/eval entries.

{_render_policy_table(policy_rows)}

## 7. Output Inventory

{_render_output_table(output_rows)}

## 8. Current Takeaways

{_render_findings_en(policy_rows)}

## 9. Task Plan

{_render_task_plan_en()}

## 10. FAST Analysis

{_render_fast_analysis_en()}

## 11. Next Steps

{_render_next_steps_en()}

## 12. Reproduction

```bash
scripts/launch_detached_milestone_pipeline.sh
tail -f logs/detached/milestone_pipeline_*.log
```

Generate reports only:

```bash
python3 scripts/generate_milestone_report.py
```
"""


if __name__ == "__main__":
    main()
