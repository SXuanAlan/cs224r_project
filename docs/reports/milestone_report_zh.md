# 动作表示 Milestone 实验报告

生成时间：2026-05-23 10:42:46

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

| Task | Status | Chunks | Action Dim | K=4 MSE | K=8 MSE | K=8 Smoothness | K=8 High-Energy | Metrics |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Lift | done | 6666 | 7 | 0.009255 | 0.003718 | 0.07214 | 0.01501 | outputs/analysis/frequency_diagnostic/metrics/lift_ph_frequency_diagnostic_20260522_230314.json |
| Can | done | 20207 | 7 | 0.007126 | 0.003159 | 0.05361 | 0.01236 | outputs/analysis/can_frequency_diagnostic/metrics/can_ph_frequency_diagnostic_20260522_230315.json |
| Push-T | done | 22560 | 2 | 0.000334 | 4.624e-05 | 0.002633 | 0.0004829 | outputs/analysis/pusht_frequency_diagnostic/metrics/pusht_frequency_diagnostic_20260522_230317.json |

## 6. Train / Eval Results

当前完成 `6/6` 个 policy train/eval 条目。

| Task | Method | Status | Best Epoch | Val Action MSE | Smoothness | Delta MSE | Rollouts | Success | Videos | Mean Return | Mean Steps | Metrics |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Lift | Raw UNet FM | done | 360 | 0.06343 | 0.1385 | 0.03707 | 50 | 1 | 50 | 1 | 52.04 | outputs/eval/lift_ph_fm_raw_20260522_230319/env/metrics.json |
| Lift | DCT Low-Frequency UNet FM | done | 910 | 0.06686 | 0.07163 | 0.02709 | 50 | 1 | 5 | 1 | 56.58 | outputs/eval/lift_ph_fm_dct_lowfreq_k8_20260522_231606/env/metrics.json |
| Can | Raw UNet FM | done | 920 | 0.05077 | 0.1095 | 0.03006 | 50 | 0.94 | 5 | 0.94 | 153.4 | outputs/eval/can_ph_fm_raw_20260522_232714/env/metrics.json |
| Can | DCT Low-Frequency UNet FM | done | 900 | 0.04716 | 0.05004 | 0.0214 | 50 | 0.62 | 5 | 0.62 | 240.7 | outputs/eval/can_ph_fm_dct_lowfreq_k8_20260522_235531/env/metrics.json |
| Push-T | Raw UNet FM | done | 480 | 0.02121 | 0.002452 | 0.001281 | 50 | 0.86 | 5 | 70.63 | 170.5 | outputs/eval/pusht_fm_raw_20260523_002454/env/metrics.json |
| Push-T | DCT Low-Frequency UNet FM | done | 560 | 0.02063 | 0.002258 | 0.00121 | 50 | 0.88 | 5 | 93.7 | 194.6 | outputs/eval/pusht_fm_dct_lowfreq_k8_20260523_005153/env/metrics.json |

## 7. Output 分类

| Output Type | Count | Pattern | Recent Examples |
|---|---:|---|---|
| Frequency diagnostics | 5 | `outputs/analysis/*/metrics/*.json` | outputs/analysis/frequency_diagnostic/metrics/lift_ph_frequency_diagnostic_20260522_230314.json<br>outputs/analysis/pusht_frequency_diagnostic/metrics/pusht_frequency_diagnostic_20260522_225714.json<br>outputs/analysis/pusht_frequency_diagnostic/metrics/pusht_frequency_diagnostic_20260522_230317.json |
| Training checkpoints | 10 | `outputs/train/*/checkpoints/*/best.pt` | outputs/train/raw_fm/checkpoints/lift_ph_fm_raw_20260522_230319/best.pt<br>outputs/train/smoke/checkpoints/lift_ph_fm_raw_smoke_20260522_223231/best.pt<br>outputs/train/smoke/checkpoints/lift_ph_fm_raw_smoke_20260522_223435/best.pt |
| Training metrics | 10 | `outputs/train/*/metrics/*.json` | outputs/train/raw_fm/metrics/lift_ph_fm_raw_20260522_230319.json<br>outputs/train/smoke/metrics/lift_ph_fm_raw_smoke_20260522_223231.json<br>outputs/train/smoke/metrics/lift_ph_fm_raw_smoke_20260522_223435.json |
| Rollout metrics | 10 | `outputs/eval/*/*/metrics.json` | outputs/eval/lift_ph_fm_raw_20260522_230319/env/metrics.json<br>outputs/eval/pusht_fm_dct_lowfreq_k8_20260523_005153/env/metrics.json<br>outputs/eval/pusht_fm_raw_20260523_002454/env/metrics.json |
| Rollout JSON | 10 | `outputs/eval/*/*/simulation_rollout.json` | outputs/eval/lift_ph_fm_raw_20260522_230319/env/simulation_rollout.json<br>outputs/eval/pusht_fm_dct_lowfreq_k8_20260523_005153/env/simulation_rollout.json<br>outputs/eval/pusht_fm_raw_20260523_002454/env/simulation_rollout.json |
| Simulation videos | 115 | `outputs/eval/*/*/*.mp4` | outputs/eval/pusht_fm_raw_20260523_002454/env/simulation_rollout_env_002.mp4<br>outputs/eval/pusht_fm_raw_20260523_002454/env/simulation_rollout_env_003.mp4<br>outputs/eval/pusht_fm_raw_20260523_002454/env/simulation_rollout_env_004.mp4 |
| Pipeline logs | 21 | `logs/pipeline/*/*.log` | logs/pipeline/20260522_230308/train_lift_raw.log<br>logs/pipeline/20260522_230308/train_pusht_dct.log<br>logs/pipeline/20260522_230308/train_pusht_raw.log |
| Detached logs | 3 | `logs/detached/*.log` | logs/detached/fm_raw_20260522_222126.log<br>logs/detached/fm_raw_20260522_223449.log<br>logs/detached/milestone_pipeline_20260522_230308.log |

## 8. 当前结论

- Lift: raw success=1, DCT success=1; DCT smoothness change=-48.3%, action MSE change=+5.4%.
- Can: raw success=0.94, DCT success=0.62; DCT smoothness change=-54.3%, action MSE change=-7.1%.
- Push-T: raw success=0.86, DCT success=0.88; DCT smoothness change=-7.9%, action MSE change=-2.7%.
- Lift 上 DCT 和 raw 都达到 100% env-reset success，说明简单任务不足以区分频率瓶颈。
- Can 上 DCT 的 validation MSE 和 smoothness 都更好，但 env success 从 0.94 降到 0.62，这是最强的 evidence：低频 reconstruction 好不等于 closed-loop manipulation 好。
- Push-T 上 DCT 与 raw 表现相当甚至略好，说明 planar pushing state benchmark 对低频动作块更友好。
- 下一步应该训练 adaptive gate / high-frequency correction：在 Lift/Push-T 保留低频平滑优势，在 Can 这种需要 precision 的阶段恢复高频。

## 9. Task 计划

任务计划按证据类型划分，而不是只堆更多 benchmark：

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
   - DROID filtered subset: 真实机器人 `insert / plug / adjust / align` 片段的 offline representation diagnostic。

## 10. FAST 对比分析

参考：FAST: Efficient Action Tokenization for Vision-Language-Action Models, arXiv:2501.09747, https://arxiv.org/abs/2501.09747

FAST 和本 milestone 的 Low-Frequency DCT baseline 都使用 DCT，但两者的 bottleneck 不同：

1. 当前 Low-Frequency DCT FM 是固定低通：只保留前 `K=8` 个 DCT temporal coefficients，所有 `k>=K` 的高频系数都被置零。
2. FAST 更接近压缩式 tokenization：对完整动作序列做 DCT，然后量化频域系数，并保留量化后非零的稀疏系数。高频系数如果幅值足够大，仍然可以被保留。
3. 因此，Can 上固定低通 DCT success 从 raw 的 0.94 降到 0.62，并不反驳 FAST。它说明 fixed low-pass bottleneck 会过度平滑 precision-sensitive manipulation，而 FAST-style sparse full-spectrum DCT 或 adaptive high-frequency correction 正是为了解决这个问题。

当前结果和 FAST 的预期关系是：低频结构确实能降低 action variation；但在 Can 这种需要抓取、对齐和短时修正的任务上，不能无条件丢弃高频。下一步应该比较 fixed low-pass DCT 与 sparse full-spectrum DCT，在相同 compression ratio 下观察 success、smoothness 和 delta MSE 的变化。

## 11. 下一步计划

下一步把以下方法和数据集扩展加入计划和实验路线：

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

所有新方法都应尽量复用当前 pipeline：UNet Flow Matching backbone、normalization、W&B logging、统一 `metrics.json`。可执行 benchmark 继续保存 `env-reset` rollout 和 simulation video；DROID 保存 offline metrics、action reconstruction plots 和 dataset trajectory visualization。

## 12. 复现实验命令

```bash
scripts/launch_detached_milestone_pipeline.sh
tail -f logs/detached/milestone_pipeline_*.log
```

单独生成报告：

```bash
python3 scripts/generate_milestone_report.py
```
