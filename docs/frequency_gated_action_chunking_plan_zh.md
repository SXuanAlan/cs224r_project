# 生成式机器人策略的动作表示项目计划

## 1. 项目动机

本项目研究机器人模仿学习中的动作表示问题，而不是只研究某一种固定的 action chunking 或 tokenizer。当前实现使用 action chunking 作为统一实验接口：策略一次预测未来一段连续动作序列。

```text
A_t = [a_t, a_{t+1}, ..., a_{t+H-1}]
```

Raw action chunk 表示简单直接，也是 diffusion policy、flow matching policy 等生成式策略常用的输出形式。但是，机械臂操作任务中的动作序列往往同时包含两类时间结构：

1. 低频、平滑的运动趋势，例如 reaching、lifting、transporting。
2. 高频、局部的修正信号，例如 grasping、alignment、gripper command transition，以及可能和接触相关的短时修正。

如果模型始终预测完整频率的原始动作表示，可能会带来不必要的 jitter；如果只保留低频表示，又可能在抓取、对齐和 gripper command transition 附近损失精度。因此，本项目的核心问题是：

```text
Can a generative robot policy use compact, smooth action representations
while preserving high-frequency corrections when precision is needed?
```

我们将从 frequency-structured action representation 开始：先将 action chunk 沿时间维做 DCT 分解，再将其拆成低频计划和高频修正。Adaptive frequency gating、sparse DCT 和 wavelet 都可以被看作这个更大问题下的候选动作表示。也就是说，action chunking 是当前训练和评估接口，tokenizer 是后续可能扩展的表示形式，但项目主线是 action representation。

第一阶段的目标不再只是离线频率分析，而是完成一个可以复现的 milestone train/eval pipeline。具体来说，milestone 必须在 Lift-PH low_dim、更难的 Can-PH low_dim，以及 Push-T state benchmark 上完成 frequency diagnostic、Raw UNet Flow Matching baseline、Low-Frequency DCT UNet Flow Matching baseline，以及标准 simulation rollout evaluation。Lift 用作 sanity check；Can 是 robomimic 里的主要完整任务；Push-T 是额外的外部 benchmark，用于检验这个频率方法是否也适用于非 robosuite 的平面 pushing task。

## 2. 项目目标

项目的最终目标是比较多类动作表示在生成式机器人策略中的效果。当前 milestone 先比较三类连续 action representation：

1. Raw Continuous Action Chunk Flow Matching
   - 作为 baseline。
   - 直接生成原始未来动作块。

2. Low-Frequency DCT Flow Matching
   - 生成低频 DCT 系数。
   - 通过 IDCT 解码回动作空间。
   - 预期动作更平滑，但可能损失精细操作能力。

3. Adaptive Frequency-Gated DCT Flow Matching
   - 生成低频计划和高频修正。
   - 学习或估计一个 gate 控制高频修正强度。
   - 目标是在 accuracy、smoothness、boundary jerk 之间取得更好的 trade-off。

后续可以继续加入 FAST-style sparse DCT 和 wavelet 等 representation。这样即使最终不做完整离散 tokenizer，项目标题和贡献也仍然成立：我们是在研究动作表示如何影响生成式 imitation policy 的精度、平滑性和执行稳定性。

Milestone 阶段必须完成：

1. Lift-PH low_dim、Can-PH low_dim 和 Push-T state 的 frequency diagnostic。
2. Raw temporal-UNet Flow Matching 的训练、checkpoint、wandb logging 和 rollout evaluation。
3. Low-Frequency DCT temporal-UNet Flow Matching 的训练、checkpoint、wandb logging 和 rollout evaluation。
4. 两类 rollout protocol：
   - `env-reset`: 正式指标，使用 `env.reset()` 随机初始状态，报告 success rate。
   - `dataset-reset`: 视频和 debugging，使用 demonstration 初始 simulator state。
5. 每个 checkpoint 都生成统一 metrics JSON，包含 validation action MSE、smoothness、delta action MSE、simulation success rate、mean return、mean steps。

Push-T 使用独立 adapter：数据来自 Diffusion Policy 的 zarr replay buffer，环境来自 `gym-pusht`。它不走 robomimic HDF5 loader，也不使用 robosuite rollout script。Adaptive gated DCT policy 可以作为 final project 的扩展；如果 milestone 时间允许，可以先实现 oracle gate diagnostic，但不能替代完整 train/eval pipeline。

DROID 作为 final-project scale-up 数据集加入计划。DROID 是 in-the-wild robot manipulation dataset，官方论文/主页报告约 76k demonstration trajectories、350 小时 interaction data，覆盖数百个真实场景和数十个任务。DROID 的作用不是替代 robomimic / Push-T 的 closed-loop simulation evaluation，而是检验频率 tokenizer 是否能扩展到真实机器人、多场景、多任务动作数据。DROID 阶段默认先做 offline frequency/tokenizer analysis 和 tokenizer / gate 预训练；除非接入真实机器人或可复现 evaluator，否则不报告 simulation success。

## 3. 方法原理

### 3.1 Action Chunk 表示

每个时间步的原始动作记为：

```text
a_t in R^d
```

其中 `d` 是 action dimension。默认 action representation 使用：

```text
rel_pos: 3D
rel_rot_6d: 6D
gripper: 1D
```

因此：

```text
d = 10
```

Action chunk 是从当前时间 `t` 开始的一段未来动作：

```text
A_t = [a_t, a_{t+1}, ..., a_{t+H-1}]
```

如果 horizon 是 `H=16`，则：

```text
A_t shape = [H, d] = [16, 10]
```

这里第一维是时间，第二维是动作维度。

### 3.2 为什么 DCT 后 Shape 仍然是 `[H, d]`

本项目对 action chunk 沿时间维做 DCT：

```text
Z_t = DCT(A_t, axis=time)
```

DCT 不改变 tensor shape，而是把时间维的语义从 time index 转换为 frequency index。因此：

```text
A_t shape = [H, d]
Z_t shape = [H, d]
```

区别在于含义不同：

```text
A_t[i, j] = 第 i 个未来时间步、第 j 个 action dimension 的动作值
Z_t[k, j] = 第 j 个 action dimension 在第 k 个 temporal frequency 上的 DCT 系数
```

对每个 action dimension，DCT 都会把长度为 `H` 的时间序列变成长度为 `H` 的频率系数序列。低 index 的 DCT 系数表示慢变化趋势，高 index 的 DCT 系数表示快速变化或局部修正。

### 3.3 IDCT 的含义

IDCT 是 inverse DCT，也就是 DCT 的逆变换。DCT 把动作序列从时间域转换到频率域，IDCT 则把频率系数转换回时间域动作序列：

```text
Z_t = DCT(A_t)
A_t = IDCT(Z_t)
```

在使用正交归一化 DCT 时，DCT 和 IDCT 理论上是无损互逆的。因此，如果保留全部频率系数：

```text
Z_full = Z_t
A_recon = IDCT(Z_full)
```

则：

```text
A_recon ≈ A_t
```

其中误差只来自数值精度。

本项目使用 IDCT 的目的不是改变动作语义，而是把修改后的频率表示解码回原始 action space，方便计算 MSE、smoothness，并最终用于机器人执行。例如：

```text
原始动作块 A_t
  -> DCT 得到频率系数 Z_t
  -> 删除或缩放部分高频系数
  -> IDCT 得到新的动作块 A_hat
```

低频重建和 adaptive gate 都依赖 IDCT：

```text
A_low = IDCT([Z_low, 0])
A_gated = IDCT([Z_low, alpha * Z_high])
```

因此，DCT/IDCT 提供的是一种 action representation 的变换方式：模型或分析可以在频率域操作动作块，但最终评价和执行仍然回到时间域 action chunk。

### 3.4 `K` 的含义

`K` 表示 DCT 后保留的低频系数数量：

```text
Z_low = Z_t[0:K]
Z_high = Z_t[K:H]
```

例如 `H=16`：

```text
K=2: 只保留最粗的低频趋势
K=4: 保留少量低频成分
K=8: 保留一半频率，通常是平滑和精度的折中
K=12: 接近完整动作
K=16: 等于 H，不做频率截断，理论上可完全重建
```

低频重建的做法是把高频系数置零：

```text
Z_hat[0:K] = Z_t[0:K]
Z_hat[K:H] = 0
A_low = IDCT(Z_hat)
```

如果 `A_low` 和原始 `A_t` 的 MSE 很低，说明这个 chunk 的主要动作信息可以由低频成分解释。如果 MSE 很高，说明高频系数携带了重要信息。

### 3.5 高频残差和 Adaptive Gate

完整动作可以写成低频计划和高频修正：

```text
A_t = IDCT([Z_low, Z_high])
```

Adaptive gate 的想法是只在需要时使用高频修正：

```text
A_hat = IDCT([Z_low, alpha * Z_high])
```

其中：

```text
alpha in [0, 1]
```

直觉：

```text
alpha = 0: 只使用低频计划，动作更平滑
alpha = 1: 使用完整高频修正，动作更精确但可能更抖
0 < alpha < 1: 在平滑和精度之间折中
```

最终 policy 中，`alpha` 可以由当前 observation 预测：

```text
alpha = g(o_t)
```

第一阶段不会把 `alpha*` 当作真实标签，而是把它作为诊断量：如果 gripper command transition 附近的 chunk 在多个 `lambda` 下都更需要高频修正，就说明 adaptive gate 有合理动机。

### 3.6 与 Flow Matching Baseline 的关系

Raw Action Chunking Flow Matching baseline 直接生成原始动作块：

```text
model(o_t) -> A_t
```

DCT 版本则生成频率域表示：

```text
model(o_t) -> Z_t
A_t = IDCT(Z_t)
```

Adaptive gated 版本进一步把频率域拆成低频、高频和 gate：

```text
model(o_t) -> Z_low, Z_high, alpha
A_hat = IDCT([Z_low, alpha * Z_high])
```

因此三种方法共享相同的 data loader、normalization、evaluation metrics 和 receding-horizon evaluation，只是 action chunk 的生成空间不同。

## 4. 环境配置

### 4.1 Python 环境

使用已有的 `cs224r` conda 环境：

```bash
conda activate cs224r
```

建议安装本地 robosuite 和 robomimic：

```bash
python -m pip install -e robosuite
python -m pip install -e robomimic
```

项目额外依赖：

```bash
python -m pip install numpy h5py scipy matplotlib pandas pyyaml omegaconf tqdm
```

如果训练 Flow Matching policy，还需要 PyTorch。robomimic 的依赖中已经包含 `torch`，但可以显式确认：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

### 4.2 数据集安装

本项目主要使用 robomimic low-dimensional imitation datasets。默认建议下载完整 simulation low_dim 数据：

```bash
python robomimic/robomimic/scripts/download_datasets.py \
  --tasks sim \
  --dataset_types all \
  --hdf5_types low_dim
```

这会覆盖 Lift、Can、Square、Transport、Tool Hang 等 simulation tasks 的 low_dim 数据。第一阶段 robomimic 任务优先使用：

```text
Lift-PH low_dim
Can-PH low_dim
```

Lift 用于 sanity check，Can 用于验证更复杂抓取、对齐和 contact-relevant phase 中是否有更明显的高频信号。注意：本项目第一阶段不直接观测物理接触，而是使用 gripper command transition 作为 contact-relevant phase 的 proxy。

Push-T 需要额外依赖和数据：

```bash
scripts/setup_pusht_deps.sh
python scripts/download_pusht_dataset.py
```

Push-T 数据默认放在：

```text
data/pusht/pusht_cchi_v7_replay.zarr
```

Push-T action 是二维目标位置 `[x, y]`，observation 默认是 5D state `[agent_x, agent_y, block_x, block_y, block_angle]`。Push-T 没有 gripper transition，因此频率分析主要按 phase、action smoothness 和 rollout success 来解释，不使用 gripper proxy。

DROID 作为后续 scale-up 数据集接入：

```text
Official page: https://droid-dataset.github.io/
Paper: https://arxiv.org/abs/2403.12945
Planned local path: data/droid/
```

DROID 接入不走 robomimic HDF5 loader，也没有可以直接 `env-reset` 的仿真环境。计划新增 `DROIDChunkDataset` 或 RLDS / HuggingFace adapter，把真实机器人 trajectory 映射到统一 action chunk 接口：

```text
observation: image / proprio / language metadata, first stage can use proprio-only or cached visual features
action: canonical end-effector delta pose + gripper
rotation: convert to rot_6d or another continuous representation
chunk: [H, d], same DCT / tokenizer interface as robomimic and Push-T
```

DROID 的第一版实验只使用 offline 指标：reconstruction MSE、smoothness、delta MSE、compression ratio、high-frequency energy ratio、per-action-group spectra，以及 dataset-video qualitative visualization。DROID 不用于 milestone 的 simulation success table；它用于 final project 的 real-world data scale-up、tokenizer pretraining、以及跨任务频率结构验证。

精密插孔、灵巧手指和小幅接触调整任务的数据集候选：

```text
ManiSkill: https://maniskill.readthedocs.io/
D4RL / Adroit: https://github.com/Farama-Foundation/d4rl/wiki/Tasks
Gymnasium-Robotics Adroit: https://robotics.farama.org/main/envs/adroit_hand/adroit_door/
RoboHive datasets: https://github.com/vikashplus/robohive/wiki/7.-Datasets
Meta-World tasks: https://metaworld.farama.org/benchmark/task_descriptions/
```

优先级建议：

1. 精密插孔 / 插头类：优先使用 ManiSkill。
   - `PegInsertionSide-v1`: peg-in-hole side insertion，直接对应精密插孔和 alignment。
   - `PlugCharger-v1`: 抓取 charger 并插入 receptacle，直接对应插头 / 插孔任务。
   - ManiSkill 支持 state / RGBD observation、可下载 demonstrations、可进行 simulation rollout，是最适合加入当前 train/eval pipeline 的下一类 benchmark。

2. 当前 robomimic 可补充：Square 和 Tool Hang。
   - `Square`: nut assembly，需要抓取、对齐和放置到 peg 上，属于 precision placement / assembly。
   - `Tool Hang`: 更长程、更接近细粒度对齐和接触调整。
   - 它们和现有 robomimic loader 兼容，接入成本最低，但不是真正的插孔或灵巧手任务。

3. 灵巧手指：优先使用 D4RL / Adroit 或 RoboHive。
   - Adroit/D4RL 包含 `pen`、`door`、`hammer`、`relocate` 等 24-DoF Shadow Hand manipulation 数据集。
   - RoboHive 也包含 dexterous manipulation / Shadow Hand 相关环境和数据。
   - 这类任务 action dimension 和控制语义与 Panda gripper 完全不同，需要单独 `DexterousHandChunkDataset` adapter，不应混进当前 7-D / 10-D end-effector action pipeline。

4. 小幅接触调整：使用 ManiSkill + DROID 过滤。
   - ManiSkill 的 insertion / plug / assembly / poke / pull 等任务可提供可执行 simulation evaluator。
   - DROID 可通过 language annotations 过滤 `insert`、`plug`、`place into`、`adjust`、`align`、`push slightly`、`move a little` 等关键词，并结合 gripper close、end-effector speed / acceleration、small action delta 来定位小幅接触调整片段。
   - DROID 过滤结果必须先做数据审计，报告 episode count、action distribution、视频样例；不能假设所有关键词都对应真实接触。

如果磁盘空间有限，可以先只下载：

```bash
python robomimic/robomimic/scripts/download_datasets.py \
  --tasks lift can \
  --dataset_types ph \
  --hdf5_types low_dim
```

### 4.3 可复现实验要求

每次实验需要保存：

```text
logs/<run_name>/
  config.yaml
  metrics.json
  metrics.csv
  train.log
  git_status.txt
  checkpoints/

outputs/
  figures/
  metrics/
  checkpoints/
```

所有路径、数据集、模型、训练参数都通过 config 控制，避免硬编码。

## 5. 代码架构

项目不使用 `milestone/` 这种临时结构，而是直接使用 final project 架构：

```text
cs224r_project/
  configs/
    data/
      robomimic_all_lowdim.yaml
      robomimic_lift_lowdim.yaml
      robomimic_can_lowdim.yaml

    analysis/
      frequency_diagnostic.yaml

    train/
      fm_action_chunk_raw.yaml
      fm_dct_lowfreq.yaml
      fm_dct_gated.yaml

  scripts/
    download_robomimic_datasets.py
    inspect_dataset.py
    run_frequency_analysis.py
    train_flow_matching.py
    eval_policy.py
    plot_results.py

  src/
    fgac/
      __init__.py

      data/
        robomimic_hdf5.py
        chunk_dataset.py
        obs_utils.py
        normalization.py
        splits.py

      transforms/
        dct.py
        frequency_split.py
        gates.py

      models/
        time_embedding.py
        mlp.py
        flow_matching.py
        raw_action_chunk_fm.py
        dct_action_chunk_fm.py
        gated_dct_fm.py

      training/
        trainer.py
        losses.py
        checkpointing.py
        metrics.py

      analysis/
        frequency_metrics.py
        transition_events.py
        boundary_jerk.py
        plotting.py

      utils/
        config.py
        logging.py
        seed.py
        device.py
        paths.py

  logs/
  outputs/
  tests/
```

### 5.1 Data 模块

Data layer 负责从 robomimic HDF5 文件中读取 demonstrations，并构造 action chunks。

关键要求：

1. HDF5 lazy loading
   - 避免在 dataloader worker 之间 pickle HDF5 handle。

2. 不跨 episode 构造 chunk
   - 对每个 demo 单独构造 index。

3. Demo-level split
   - train / val split 必须按 demo 划分，避免相邻 chunk 泄漏。

4. Config-driven observation keys
   - low_dim obs 可能包括 `robot0_eef_pos`、`robot0_eef_quat`、`robot0_gripper_qpos`、`object` 等。
   - 具体 keys 应由 config 指定。

5. Action representation
   - 首选动作表示为 `action_dict/rel_pos`、`action_dict/rel_rot_6d`、`action_dict/gripper` 拼接得到的 10D action。
   - 选择 6D rotation 是为了避免 axis-angle 表示中的不连续性人为制造高频能量。
   - 如果数据集中没有 `action_dict`，先运行 robomimic 的 `extract_action_dict.py` 生成；legacy `actions` 只作为 sanity-check fallback。

6. Normalization
   - 默认使用 per-dimension min-max normalization，将每个 action dimension 映射到 `[-1, 1]`。
   - min / max 只用 training split 统计，避免 validation leakage。
   - obs normalization 默认使用 per-dimension z-score，也只用 training split 统计。
   - 频率能量和 MSE 主要在 normalized action space 中计算；raw action space 结果可作为附录。
   - z-score action normalization 可以作为 ablation，但不是默认设置。

### 5.2 Transform 模块

`transforms/dct.py` 提供：

```python
dct_time(x)
idct_time(z)
```

输入 shape：

```text
[batch, horizon, action_dim]
```

DCT 沿 horizon 维进行。

`frequency_split.py` 提供：

```python
split_low_high(z, k)
low_reconstruct(actions, k)
high_energy_ratio(z, k)
```

`gates.py` 提供：

```python
oracle_alpha(actions, z, k, alphas, lambda_alpha)
```

后续 gated model 使用同一套接口。

## 6. 实验设计

### 6.1 Experiment A: Frequency Decomposition Diagnostic

目的：验证动作块中的低频和高频结构是否有意义。

输入：

```text
Dataset: robomimic Lift-PH low_dim, robomimic Can-PH low_dim, Push-T state
Horizon H: 16
Stride: 1
K values: [2, 4, 8, 12, 16]
```

DROID 扩展阶段使用同一套 Experiment A offline diagnostic，但不进入 simulation rollout 表：

```text
Dataset: DROID subset/full split
Horizon H: 16 or dataset-control-rate-adjusted horizon
Stride: 1 for coverage, stride=H for non-overlapping sanity check
Metrics: reconstruction MSE, smoothness, delta MSE, compression ratio, high-frequency energy
```

说明：

1. `K=16` 等于 `K=H`，是无频率截断的 trivial full-reconstruction endpoint，MSE 应该为 0 或接近数值误差。
2. `stride=1` 用于覆盖所有 possible chunks，但相邻 chunks 高度重叠，不能把它们当作独立样本来报告过度自信的 error bar。
3. 聚合统计默认报告 mean / median；如果需要 confidence interval，使用 demo-level bootstrap。
4. 额外运行 `stride=H` 的 non-overlapping sanity check，验证结论不是由重叠 chunk 造成。

流程：

1. 读取每条 demonstration 的 native actions。
2. 构造 action chunks。
3. 对每个 chunk 沿时间维做 DCT。
4. 对每个 `K` 只保留前 `K` 个频率。
5. 用 IDCT 重建 low-frequency action chunk。
6. 计算 reconstruction、smoothness、high-frequency energy。
7. 分别报告 aggregate spectrum 和 per-action-group spectrum。

预期现象：

1. `K` 增大时 reconstruction MSE 下降。
2. 低频重建动作比 raw action 更 smooth。
3. 高频能量在 gripper command transition 附近更高。

Per-action-group analysis 至少分成三组：

```text
translation: rel_pos
rotation: rel_rot_6d
gripper: gripper
```

这样可以区分高频能量到底来自 gripper channel，还是也影响 arm motion。

### 6.2 Experiment B: Gripper Transition Conditioned Analysis

目的：验证高频修正是否和 gripper command transition 相关。这里的 gripper transition 只是 contact-relevant phase 的 proxy，不等同于真实物理接触。

定义 gripper transition：

```text
|a_gripper[t+1] - a_gripper[t]| > tau
```

阈值选择：

1. 先用 `inspect_dataset.py` 检查 gripper action distribution。
2. 如果 gripper action 近似二值，默认使用 `|delta_gripper| > 1.0`。
3. 如果 gripper action 不是二值，再使用 90 或 95 percentile 作为 fallback。

将 chunk 分成：

```text
near transition
away from transition
```

比较：

1. high-frequency energy ratio
2. oracle alpha and relative MSE reduction
3. reconstruction MSE
4. smoothness

预期现象：

```text
near-transition chunks should have higher high-frequency energy,
higher relative MSE reduction from high-frequency coefficients,
and possibly higher oracle alpha under a range of lambda values.
```

如果 Lift 上现象不明显，应在 Can 上重复实验。Push-T 没有 gripper channel，因此不跑 gripper-transition-conditioned split；Push-T 使用 task phase / progress-conditioned smoothness 和 rollout success 作为补充分析。

DROID 上的 transition analysis 应该显式区分 gripper command transition 和真实接触。第一版只使用 gripper command、end-effector speed / acceleration、action delta percentile、以及 language/task metadata 作为 phase proxy；除非数据中有可靠 force/contact annotation，否则不把这些 proxy 称为 physical contact。

新增任务族的 Experiment A 计划：

1. ManiSkill precision insertion / plug-in
   - `PegInsertionSide-v1` 和 `PlugCharger-v1` 作为首选。
   - 运行同样的 DCT / sparse DCT / wavelet offline diagnostic。
   - 因为有 simulation environment，后续可加入 Flow Matching train/eval 和 rollout success。

2. robomimic Square / Tool Hang
   - 作为当前 robomimic pipeline 的低成本扩展。
   - 重点比较 low-pass DCT 是否在 precision assembly 阶段失败，以及 FAST-style sparse DCT 是否恢复 success。

3. D4RL / Adroit 或 RoboHive dexterous hand
   - 先只跑 offline chunk frequency/tokenizer diagnostic。
   - 新增 hand-action normalization、per-joint spectra、finger-group smoothness，以及 high-dimensional action chunk reconstruction。
   - closed-loop evaluation 只有在 hand environment adapter 稳定后再加入。

4. DROID contact-adjustment filtered subset
   - 先按 language keyword 和 trajectory statistics 生成候选 subset。
   - 人工抽查视频，确认是否真的包含插入、对齐、小幅接触调整。
   - 报告 subset size、关键词分布、动作幅度分布和高频能量分布。

### 6.2.1 用于验证 Reconstruction Smoothness 的任务计划

为了证明 reconstruction smoothness 不是只在离线指标上变好，而是对控制有实际意义，任务选择要分成正例、sanity check 和反例：

1. Positive evidence: Push-T
   - Push-T 是当前最干净的 smoothness 任务：连续平面 pushing 需要稳定、低抖动动作。
   - 当前结果中 DCT 表示降低 action variation，同时 success 和 return 不下降，适合证明 smooth reconstructed actions 可以帮助或至少不伤害平滑控制。
   - 后续可以加入 harder Push-T variants，例如更窄目标、更长 horizon、更复杂初始分布或障碍物。

2. Sanity evidence: Lift
   - Lift 主要由 reaching 和 lifting 组成，天然偏低频。
   - DCT 和 raw 都能成功，说明 smooth reconstruction 不会破坏简单 manipulation。
   - 但 Lift 太简单，不能单独作为 smoothness 有用的强证据。

3. Trade-off / negative evidence: Can
   - Can 上 DCT smoothness 和 validation MSE 更好，但 closed-loop success 明显下降。
   - 这个任务用来证明 smoothness alone 不够，precision-sensitive phase 需要保留高频信息。
   - 它是 sparse DCT、adaptive gate 和 wavelet 的主要动机。

4. Future stress tests
   - ManiSkill `PegInsertionSide-v1` / `PlugCharger-v1`: 验证插孔、插头和小幅接触调整中 smoothness 与 precision 的冲突。
   - robomimic Square / Tool Hang: 低成本验证 precision assembly 阶段是否需要高频修正。
   - DROID filtered subset: 离线验证真实机器人 `insert / plug / adjust / align` 片段中，高频保留是否比固定低频重建更可靠。

报告时应明确区分：

```text
Push-T: smoothness-positive task
Lift: smoothness sanity-check task
Can / insertion / contact-adjustment: smoothness-precision trade-off task
```

### 6.3 Experiment C: Raw Action Chunking Flow Matching Baseline

这是普通 action chunking baseline。

目标：

```text
condition: current observation o_t
target: raw action chunk A_t = [a_t, ..., a_{t+H-1}]
```

Flow Matching 训练：

```text
x0 ~ N(0, I)
x1 = normalized raw action chunk
t ~ Uniform(0, 1)
xt = (1 - t) x0 + t x1
target velocity = x1 - x0
model output = v_theta(xt, t, o_t)
loss = MSE(v_theta, x1 - x0)
```

模型结构使用 temporal UNet，而不是简单 MLP：

1. observation history encoder：默认 `observation_horizon=2`，将最近两帧 low_dim observation 作为 condition。
2. sinusoidal Flow Matching time embedding。
3. temporal 1D UNet：输入和输出都是 `[batch, H, action_dim]` 的 action sequence。
4. EMA weights：evaluation 和 checkpoint 默认使用 EMA 参数。

输出 shape：

```text
[batch, H, action_dim]
```

采样：

1. 从 Gaussian noise 初始化 action chunk。
2. 用 Euler integration 从 `t=0` 积分到 `t=1`。
3. 得到 normalized action chunk。
4. unnormalize 回原始 action space。

这个 baseline 是后续 DCT / gated method 的主要比较对象。Milestone 中它必须在 Lift 和 Can 上都完成训练和 rollout evaluation。

### 6.4 Experiment D: Low-Frequency DCT Flow Matching

目标从 raw action chunk 改为 DCT low-frequency coefficients。

训练目标：

```text
z = DCT(A)
target = z[:K]
```

模型生成低频系数后，用 zero-padded DCT coefficients 做 IDCT：

```text
A_hat = IDCT([z_low, 0])
```

评价时仍然在 action space 里比较：

1. decoded action MSE
2. smoothness
3. delta action MSE
4. boundary jerk

Milestone 中 DCT low-frequency FM 和 Raw FM 使用相同的 temporal UNet、observation horizon、normalization、wandb logging 和 rollout protocol，只改变训练目标空间。

预期：

```text
DCT low-frequency policy should be smoother than raw baseline,
but may lose precision near gripper command transitions or contact-relevant phases.
```

### 6.5 Experiment E: Adaptive Frequency-Gated DCT Flow Matching

目标是生成低频计划、高频修正和 gate：

```text
z_low, z_high, alpha = model(o_t)
A_hat = IDCT([z_low, alpha * z_high])
```

训练可以分两阶段：

1. Oracle gate pretraining
   - 用离线计算的 `alpha*` 作为 pseudo-label。
   - `alpha*` 不是 ground-truth gate，只是受 `lambda` 控制的诊断量和可选 warm-start target。
   - gate loss 使用 MSE 或 cross entropy。

2. End-to-end decoded action training
   - 用 decoded action MSE 和 flow matching loss 共同训练。

初期可先报告 `K=4` 和 `K=8`，训练阶段先固定 `K=8`，后续调参 `K in [4, 8, 12]`。

预期：

```text
Adaptive gated DCT policy should approach raw baseline accuracy,
while reducing unnecessary high-frequency jitter away from transitions.
```

### 6.6 FAST 对比和后续 Tokenizer 路线

FAST（Frequency-space Action Sequence Tokenization）也使用 DCT，但它和当前 milestone 的 Low-Frequency DCT baseline 不是同一个方法。当前 baseline 固定只保留前 `K=8` 个低频系数，并把所有高频系数置零；这等价于一个 hard low-pass bottleneck。FAST 的核心思想更接近 time-series compression：先对完整动作序列做 DCT，再量化频域系数，并保留量化后非零的稀疏系数。也就是说，FAST 不是简单丢弃高频，而是在压缩预算内保留幅值足够大的高频成分。因此，如果 gripper transition、alignment correction 或 contact-relevant correction 在高频 DCT 系数上有较大幅值，FAST-style tokenizer 可以保留这些信号。

这个区别对当前结果很重要：Can 上 Low-Frequency DCT 的 validation MSE 和 smoothness 更好，但 closed-loop success 明显下降。这不反驳 FAST；它反而说明固定低通 DCT 容易过度平滑，需要 sparse high-frequency retention 或 adaptive high-frequency correction。

下一阶段优先加入两个 tokenizer-inspired / representation 方向：

- 方法 1: FAST-style Sparse DCT Tokenizer
   - 对完整 DCT 系数做量化和稀疏保留，而不是固定截断到 `K=8`。
   - 先实现连续版本：保留 top-magnitude DCT coefficients 或阈值化 nonzero coefficients，并用 IDCT 解码。
   - 再考虑离散版本：quantization + BPE / vocabulary compression，用于和 VLA-style tokenizer 对齐。
   - 主要问题：在相同 compression ratio 下，是否比 fixed low-pass DCT 更好地保持 Can success，同时保留 smoothness 优势。

- 方法 4: Wavelet Tokenizer
   - 用 multi-resolution temporal basis 替代纯 DCT。
   - Wavelet 更适合表示局部突变，因此可能比 fixed low-pass DCT 更适合 gripper transition 和短时 correction。
   - 主要问题：是否能在 Can 这类 precision-sensitive task 上保留局部高频，同时在 reaching / transporting 阶段维持平滑。

这两个方向都应复用现有 train/eval pipeline：同一个 temporal UNet Flow Matching backbone、同一套 normalization、同一套 `env-reset` rollout、同一套 metrics JSON 和视频保存逻辑。这样新 representation 的收益可以直接和 Raw FM、Low-Frequency DCT FM 比较。

### 6.7 DROID 真实世界数据集 Scale-Up

DROID 作为 robomimic 和 Push-T pipeline 稳定后的真实世界 scale-up 阶段。它的直接目标不是 closed-loop rollout success，因为 DROID 是真实机器人数据集，没有本地 simulator reset protocol。DROID 要回答的问题是：从 Lift / Can / Push-T 得到的 tokenizer 设计，是否仍然适用于多场景、多任务、真实机器人轨迹。

计划中的 DROID 使用方式：

1. 构建 DROID action-chunk adapter。
   - 读取 RLDS / HuggingFace / local DROID trajectories。
   - 将 action 映射到统一表示，例如 end-effector delta pose + gripper。
   - 将 rotation 转成 `rot_6d` 或其他连续表示，避免 discontinuity。
   - 将 trajectory resample 或 window 成 `[H, d]` chunks。

2. 运行 offline tokenizer diagnostics。
   - Fixed low-pass DCT reconstruction。
   - FAST-style sparse DCT reconstruction，在相同 compression ratio 下比较。
   - Wavelet reconstruction，重点看局部 action jump。

3. 使用 DROID 预训练。
   - 在多样真实轨迹上预训练 sparse/wavelet representation 或 adaptive gate。
   - 在 robomimic Can、Push-T 或其他可执行 benchmark 上 fine-tune / evaluate。
   - DROID 结果报告为 offline pretraining 和 representation evidence，不报告为 simulation success。

4. 保存 qualitative artifacts。
   - Dataset trajectory visualizations。
   - Reconstructed action plots。
   - 可选视频：对比 original trajectory metadata 和 reconstructed/tokenized actions。

DROID 主要指标：

```text
reconstruction MSE
delta action MSE
smoothness
compression ratio
high-frequency energy ratio
per-action-group spectra
transition-proxy-conditioned reconstruction error
```

如果后续有真实机器人 evaluation path，可以评估 DROID-pretrained tokenizer / policy 的 task success；在此之前，DROID 在本计划中定位为 offline scale-up 和 pretraining dataset。

## 7. 评估指标

### 7.0 Rollout Protocol

Milestone 报告必须区分两种 simulation evaluation：

1. `env-reset` official evaluation
   - 使用 `env.reset()` 随机初始化环境。
   - 每个方法和任务至少 `50` rollouts；调试阶段可以先跑 `10`。
   - 这是报告 success rate、mean return、mean steps 的正式 protocol。

2. `dataset-reset` video/debug evaluation
   - 使用 dataset demonstration 的初始 simulator state。
   - 用于生成可比较视频和 debug policy behavior。
   - 不能作为最终 success rate 的唯一证据，因为它比随机 env reset 更接近训练分布。

每次 eval 需要保存：

```text
simulation_rollout.json
metrics.json
simulation_rollout_*.mp4
```

其中 `metrics.json` 至少包含 validation action MSE、smoothness、delta action MSE、success rate、mean return、mean steps。

### 7.1 Reconstruction MSE

用于 DCT diagnostic：

```text
MSE_rec(K) = mean(||A - IDCT([Z_0:K, 0])||^2)
```

解释：

1. 衡量低频系数能保留多少动作信息。
2. `K` 越大，MSE 应该越低。
3. 如果 `K=4` 或 `K=8` 已经很低，说明动作有强低频结构。

### 7.2 Smoothness

新增核心指标：

```text
S(A) = 1 / (H - 1) * sum_{i=0}^{H-2} ||a_{i+1} - a_i||_2^2
```

解释：

1. 衡量 action chunk 内部的 temporal jitter。
2. 数值越低表示动作越平滑。
3. 对 raw action、DCT reconstruction、policy prediction 都计算。

注意：

```text
Smoothness cannot be evaluated alone.
An overly constant action sequence can be smooth but inaccurate.
```

因此 smoothness 必须和 action MSE 一起看。

### 7.3 Delta Action MSE

为了避免模型过度平滑，加入 delta matching：

```text
DeltaMSE(A_hat, A)
= mean(||(A_hat[:, 1:] - A_hat[:, :-1]) - (A[:, 1:] - A[:, :-1])||^2)
```

解释：

1. 衡量预测动作变化是否匹配真实动作变化。
2. 比单纯 smoothness 更能反映 temporal dynamics。
3. 对 policy evaluation 很重要。

### 7.4 High-Frequency Energy Ratio

```text
E_high(K) = sum_{k=K}^{H-1} ||Z_k||^2 / (sum_{k=0}^{H-1} ||Z_k||^2 + eps)
```

解释：

1. 衡量动作块中有多少信息位于高频部分。
2. 可以按 task、phase、transition condition 聚合。
3. 必须同时报告 aggregate ratio 和 per-action-group ratio，避免 gripper dimension 主导所有结论。

### 7.5 Oracle Alpha

```text
alpha* = argmin_alpha [
  ||IDCT([Z_low, alpha Z_high]) - A||^2 + lambda * alpha
]
```

其中：

```text
alpha in [0, 0.25, 0.5, 0.75, 1.0]
lambda in [1e-4, 1e-3, 1e-2, 1e-1]
```

解释：

1. 当 `alpha=1` 时，DCT/IDCT 可以完整恢复原始 chunk，因此 reconstruction MSE 为 0 或接近数值误差。
2. 因此 `alpha*` 不是 ground-truth gate，而是由 high-frequency reconstruction gain 和 `lambda` penalty 共同决定的诊断量。
3. 必须对 `lambda` 做 sweep，报告 `alpha*` distribution 如何变化，避免只选择一个有利的 penalty。
4. 默认在 `K=4` 和 `K=8` 上报告 oracle analysis。
5. 如果 transition 附近在多个 `lambda` 下都有更高 `alpha*`，这才支持 adaptive gating 的设计。

同时报告更稳健的 relative MSE reduction：

```text
RelGain(alpha)
= (MSE_low - MSE_alpha) / (MSE_low + eps)
```

其中 `MSE_low` 是 `alpha=0` 的低频重建误差，`MSE_alpha` 是给定 `alpha` 后的重建误差。这个指标比单个 `alpha*` 对 `lambda` 更不敏感。

### 7.6 Boundary Jerk

模拟 receding-horizon execution：

```text
J_boundary = ||a_new[t] - a_old[t-1]||_2
```

实现方式：

1. `K_exec` 表示每次预测一个 chunk 后实际执行的 action 数量，然后重新预测；它等价于 Diffusion Policy 中的 action horizon `T_a`。
2. 每隔 `K_exec` 步切换到一个新 action chunk。
3. 比较旧 chunk 最后一个执行动作和新 chunk 第一个执行动作。

解释：

1. 衡量连续 chunk 之间的 discontinuity。
2. Raw action chunking 可能 accuracy 高但 boundary jerk 大。
3. Low-frequency 或 gated representation 可能降低 jerk。
4. 在离线 frequency diagnostic 中，boundary jerk 只能作为 reconstructed chunks 的辅助分析；它在训练 policy 后的 receding-horizon prediction 中才是主要 evaluation metric。

### 7.7 Policy Metrics

每个 policy run 至少报告：

```text
val/flow_matching_loss
val/action_mse
val/smoothness
val/delta_action_mse
val/boundary_jerk
val/gripper_transition_action_mse
val/non_transition_action_mse
val/gripper_transition_smoothness
val/non_transition_smoothness
```

可选 rollout 后报告：

```text
rollout/success_rate
rollout/return
rollout/horizon
rollout/mean_action_smoothness
```

## 8. 结果图表

第一阶段至少生成：

1. Reconstruction MSE vs K
2. Smoothness vs K
3. High-frequency energy near vs away transition
4. Per-action-group high-frequency spectra
5. Relative MSE reduction near vs away transition
6. Oracle alpha near vs away transition under lambda sweep

Flow Matching baseline 阶段生成：

1. training / validation loss curve
2. decoded action MSE curve
3. smoothness comparison
4. boundary jerk comparison
5. `env-reset` simulation success rate
6. `dataset-reset` qualitative videos

最终报告核心比较表：

```text
Task    Method                    Action MSE    Smoothness    Delta MSE    Env Success    Mean Steps
Lift    Raw UNet FM               ...
Lift    DCT Low-Frequency UNet FM ...
Can     Raw UNet FM               ...
Can     DCT Low-Frequency UNet FM ...
```

## 9. 开发顺序

建议实现顺序：

1. 创建 config、logging、path 工具。
2. 实现 dataset inspection script。
3. 实现 robust robomimic HDF5 data loader。
4. 实现 action chunk dataset 和 normalization。
5. 实现 DCT / IDCT transform。
6. 实现 frequency diagnostic metrics。
7. 检查 gripper action distribution，确定 transition threshold。
8. 跑 Lift-PH frequency analysis。
9. 跑 Can-PH frequency analysis。
10. 实现 demo-level bootstrap 和 `stride=H` sanity check。
11. 实现 temporal-UNet raw action chunking Flow Matching baseline。
12. 加入 smoothness、delta action MSE、boundary jerk validation。
13. 实现 DCT low-frequency temporal-UNet Flow Matching。
14. 添加 `env-reset` official rollout evaluation。
15. 添加 `dataset-reset` video/debug evaluation。
16. 在 Lift 上跑完整 train/eval，作为 sanity check。
17. 在 Can 上跑完整 train/eval，作为 robomimic 主要任务。
18. 在 Push-T state 上跑完整 train/eval，作为外部 pushing benchmark。
19. 汇总统一 metrics JSON 和 comparison table。
20. 实现 adaptive gated DCT Flow Matching。
21. 实现 FAST-style sparse DCT tokenizer baseline，比较 fixed low-pass DCT 和 sparse full-spectrum DCT。
22. 实现 wavelet representation baseline，测试局部高频事件的保真度。
23. 实现 DROID adapter，先跑 DROID offline frequency/tokenizer diagnostics。
24. 使用 DROID 做 sparse/wavelet representation 或 adaptive gate pretraining，再在可执行 benchmark 上 fine-tune / evaluate。
25. 接入 ManiSkill `PegInsertionSide-v1` / `PlugCharger-v1`，作为精密插孔和插头类可执行 benchmark。
26. 接入 robomimic Square / Tool Hang，作为低成本 precision assembly 扩展。
27. 接入 D4RL / Adroit 或 RoboHive dexterous hand 数据，先做 offline hand-action representation diagnostic。
28. 从 DROID language annotations 中过滤 insertion / alignment / small contact-adjustment subset，并做人审视频抽查。

## 10. 风险和备选方案

### 风险 1: Lift 高频现象不明显

解决：

1. 使用 Can-PH。
2. 把 per-action-group spectra、relative MSE reduction 和 smoothness 作为主要 diagnostic。
3. 把 Lift 作为 sanity check，而不是唯一结论来源。

### 风险 2: Gripper transition 不是 contact 的好 proxy

解决：

1. 使用 object-state features 定义 phase。
2. 使用 action acceleration 或 delta action percentile 定义 transition。
3. 对 Can/Square 任务比较更复杂的 alignment 阶段。
4. 在最终结论中只把当前结果称为 gripper command transition evidence，不把它直接等同于 physical contact。

### 风险 3: DCT low-frequency 过度平滑

解决：

1. 加入 delta action MSE。
2. 使用 gated high-frequency correction。
3. 调整 `K` 和 `lambda_alpha`。

### 风险 4: Oracle alpha 对 lambda 过于敏感

解决：

1. 对 `lambda in [1e-4, 1e-3, 1e-2, 1e-1]` 做 sweep。
2. 报告 relative MSE reduction 作为主诊断。
3. 把 `alpha*` 定位为 optional gate warm-start，而不是 ground-truth label。

### 风险 5: Flow Matching baseline 训练不稳定

解决：

1. 先 overfit one batch。
2. 降低模型复杂度。
3. 检查 action normalization。
4. 使用较小 horizon 或较少 flow integration steps。

## 11. 最终目标陈述

本项目最终希望证明：

```text
Compact action representations can capture most smooth robot motion,
but precision-sensitive phases require preserving selected high-frequency structure.
Compared with raw continuous action chunks and fixed low-frequency bottlenecks,
adaptive or sparse action representations may preserve accuracy while improving smoothness
and reducing boundary jerk.
```
