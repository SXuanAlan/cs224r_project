# 频率门控动作块生成策略项目计划

## 1. 项目动机

本项目研究机器人模仿学习中的动作表示问题。传统的 action chunking 方法通常直接预测未来一段连续的原始动作序列：

```text
A_t = [a_t, a_{t+1}, ..., a_{t+H-1}]
```

这种表示简单直接，也是 diffusion policy、flow matching policy 等生成式策略常用的输出形式。但是，机械臂操作任务中的动作序列往往同时包含两类时间结构：

1. 低频、平滑的运动趋势，例如 reaching、lifting、transporting。
2. 高频、局部的修正信号，例如 grasping、contact、alignment、gripper transition。

如果模型始终预测完整频率的原始动作块，可能会带来不必要的 jitter；如果只保留低频动作，又可能在抓取和接触阶段损失精度。因此，本项目的核心问题是：

```text
Can a generative robot policy use low-frequency action chunks for smooth motion
and selectively activate high-frequency corrections when precision is needed?
```

我们将研究一种 Frequency-Gated Action Chunking 表示：先将动作块沿时间维做 DCT 分解，再将其拆成低频计划和高频修正。策略可以选择性地使用高频部分，而不是总是生成完整频率的动作。

## 2. 项目目标

项目的最终目标是比较三类动作块生成策略：

1. Raw Action Chunking Flow Matching
   - 作为 baseline。
   - 直接生成原始未来动作块。

2. Low-Frequency DCT Flow Matching
   - 生成低频 DCT 系数。
   - 通过 IDCT 解码回动作空间。
   - 预期动作更平滑，但可能损失精细操作能力。

3. Frequency-Gated DCT Flow Matching
   - 生成低频计划和高频修正。
   - 学习或估计一个 gate 控制高频修正强度。
   - 目标是在 accuracy、smoothness、boundary jerk 之间取得更好的 trade-off。

第一阶段重点完成离线分析和 raw Flow Matching baseline。后续再扩展到 DCT low-frequency 和 gated DCT policy。

## 3. 环境配置

### 3.1 Python 环境

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

### 3.2 数据集安装

本项目主要使用 robomimic low-dimensional imitation datasets。默认建议下载完整 simulation low_dim 数据：

```bash
python robomimic/robomimic/scripts/download_datasets.py \
  --tasks sim \
  --dataset_types all \
  --hdf5_types low_dim
```

这会覆盖 Lift、Can、Square、Transport、Tool Hang 等 simulation tasks 的 low_dim 数据。第一阶段优先使用：

```text
Lift-PH low_dim
Can-PH low_dim
```

Lift 用于 sanity check，Can 用于验证更复杂接触和对齐阶段是否有更明显的高频信号。

如果磁盘空间有限，可以先只下载：

```bash
python robomimic/robomimic/scripts/download_datasets.py \
  --tasks lift can \
  --dataset_types ph \
  --hdf5_types low_dim
```

### 3.3 可复现实验要求

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

## 4. 代码架构

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

### 4.1 Data 模块

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

5. Normalization
   - action mean / std 只用 training split 统计。
   - obs mean / std 也只用 training split 统计。
   - stats 保存到 run directory。

### 4.2 Transform 模块

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

## 5. 实验设计

### 5.1 Experiment A: Frequency Decomposition Diagnostic

目的：验证动作块中的低频和高频结构是否有意义。

输入：

```text
Dataset: robomimic Lift-PH low_dim, Can-PH low_dim
Horizon H: 16
Stride: 1
K values: [2, 4, 8, 12, 16]
```

流程：

1. 读取每条 demonstration 的 native actions。
2. 构造 action chunks。
3. 对每个 chunk 沿时间维做 DCT。
4. 对每个 `K` 只保留前 `K` 个频率。
5. 用 IDCT 重建 low-frequency action chunk。
6. 计算 reconstruction、smoothness、high-frequency energy、boundary jerk。

预期现象：

1. `K` 增大时 reconstruction MSE 下降。
2. 低频重建动作比 raw action 更 smooth。
3. 高频能量在 gripper/contact-like transition 附近更高。

### 5.2 Experiment B: Gripper Transition Conditioned Analysis

目的：验证高频修正是否和抓取 / 接触阶段相关。

定义 gripper transition：

```text
|a_gripper[t+1] - a_gripper[t]| > tau
```

其中 `tau` 使用 gripper action difference 的 90 或 95 percentile。

将 chunk 分成：

```text
near transition
away from transition
```

比较：

1. high-frequency energy ratio
2. oracle alpha
3. reconstruction MSE
4. smoothness

预期现象：

```text
near transition chunks should have higher high-frequency energy
and higher oracle alpha.
```

如果 Lift 上现象不明显，应在 Can 上重复实验。

### 5.3 Experiment C: Raw Action Chunking Flow Matching Baseline

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

模型结构：

1. observation encoder MLP
2. sinusoidal time embedding
3. noisy action chunk encoder
4. velocity prediction head

输出 shape：

```text
[batch, H, action_dim]
```

采样：

1. 从 Gaussian noise 初始化 action chunk。
2. 用 Euler integration 从 `t=0` 积分到 `t=1`。
3. 得到 normalized action chunk。
4. unnormalize 回原始 action space。

这个 baseline 是后续 DCT / gated method 的主要比较对象。

### 5.4 Experiment D: Low-Frequency DCT Flow Matching

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

预期：

```text
DCT low-frequency policy should be smoother than raw baseline,
but may lose precision near transition/contact phases.
```

### 5.5 Experiment E: Frequency-Gated DCT Flow Matching

目标是生成低频计划、高频修正和 gate：

```text
z_low, z_high, alpha = model(o_t)
A_hat = IDCT([z_low, alpha * z_high])
```

训练可以分两阶段：

1. Oracle gate pretraining
   - 用离线计算的 `alpha*` 作为 pseudo-label。
   - gate loss 使用 MSE 或 cross entropy。

2. End-to-end decoded action training
   - 用 decoded action MSE 和 flow matching loss 共同训练。

初期可先固定 `K=8`，后续调参 `K in [4, 8, 12]`。

预期：

```text
Gated DCT policy should approach raw baseline accuracy,
while reducing unnecessary high-frequency jitter away from transitions.
```

## 6. Evaluation Metrics

### 6.1 Reconstruction MSE

用于 DCT diagnostic：

```text
MSE_rec(K) = mean(||A - IDCT([Z_0:K, 0])||^2)
```

解释：

1. 衡量低频系数能保留多少动作信息。
2. `K` 越大，MSE 应该越低。
3. 如果 `K=4` 或 `K=8` 已经很低，说明动作有强低频结构。

### 6.2 Smoothness

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

### 6.3 Delta Action MSE

为了避免模型过度平滑，加入 delta matching：

```text
DeltaMSE(A_hat, A)
= mean(||(A_hat[:, 1:] - A_hat[:, :-1]) - (A[:, 1:] - A[:, :-1])||^2)
```

解释：

1. 衡量预测动作变化是否匹配真实动作变化。
2. 比单纯 smoothness 更能反映 temporal dynamics。
3. 对 policy evaluation 很重要。

### 6.4 High-Frequency Energy Ratio

```text
E_high(K) = sum_{k=K}^{H-1} ||Z_k||^2 / (sum_{k=0}^{H-1} ||Z_k||^2 + eps)
```

解释：

1. 衡量动作块中有多少信息位于高频部分。
2. 可以按 task、phase、transition condition 聚合。

### 6.5 Oracle Alpha

```text
alpha* = argmin_alpha [
  ||IDCT([Z_low, alpha Z_high]) - A||^2 + lambda * alpha
]
```

其中：

```text
alpha in [0, 0.25, 0.5, 0.75, 1.0]
```

解释：

1. `alpha*` 表示该 chunk 是否真的需要高频修正。
2. 如果 transition 附近 `alpha*` 更高，说明 adaptive gating 有意义。

### 6.6 Boundary Jerk

模拟 receding-horizon execution：

```text
J_boundary = ||a_new[t] - a_old[t-1]||_2
```

实现方式：

1. 每隔 `K_exec` 步切换到一个新 action chunk。
2. 比较旧 chunk 最后一个执行动作和新 chunk 第一个执行动作。

解释：

1. 衡量连续 chunk 之间的 discontinuity。
2. Raw action chunking 可能 accuracy 高但 boundary jerk 大。
3. Low-frequency 或 gated representation 可能降低 jerk。

### 6.7 Policy Metrics

每个 policy run 至少报告：

```text
val/flow_matching_loss
val/action_mse
val/smoothness
val/delta_action_mse
val/boundary_jerk
val/gripper_transition_action_mse
val/non_transition_action_mse
```

可选 rollout 后报告：

```text
rollout/success_rate
rollout/return
rollout/horizon
rollout/mean_action_smoothness
```

## 7. 结果图表

第一阶段至少生成：

1. Reconstruction MSE vs K
2. Smoothness vs K
3. High-frequency energy near vs away transition
4. Oracle alpha near vs away transition

Flow Matching baseline 阶段生成：

1. training / validation loss curve
2. decoded action MSE curve
3. smoothness comparison
4. boundary jerk comparison

最终报告核心比较表：

```text
Method                    Action MSE    Smoothness    Delta MSE    Boundary Jerk
Raw FM                    ...
DCT Low-Frequency FM      ...
Gated DCT FM              ...
```

## 8. 开发顺序

建议实现顺序：

1. 创建 config、logging、path 工具。
2. 实现 dataset inspection script。
3. 实现 robust robomimic HDF5 data loader。
4. 实现 action chunk dataset 和 normalization。
5. 实现 DCT / IDCT transform。
6. 实现 frequency diagnostic metrics。
7. 跑 Lift-PH frequency analysis。
8. 跑 Can-PH frequency analysis。
9. 实现 raw action chunking Flow Matching baseline。
10. 加入 smoothness、delta action MSE、boundary jerk validation。
11. 实现 DCT low-frequency Flow Matching。
12. 实现 gated DCT Flow Matching。
13. 添加 rollout evaluation。

## 9. 风险和备选方案

### 风险 1: Lift 高频现象不明显

解决：

1. 使用 Can-PH。
2. 把 boundary jerk 和 smoothness 作为主要 diagnostic。
3. 把 Lift 作为 sanity check，而不是唯一结论来源。

### 风险 2: Gripper transition 不是 contact 的好 proxy

解决：

1. 使用 object-state features 定义 phase。
2. 使用 action acceleration 或 delta action percentile 定义 transition。
3. 对 Can/Square 任务比较更复杂的 alignment 阶段。

### 风险 3: DCT low-frequency 过度平滑

解决：

1. 加入 delta action MSE。
2. 使用 gated high-frequency correction。
3. 调整 `K` 和 `lambda_alpha`。

### 风险 4: Flow Matching baseline 训练不稳定

解决：

1. 先 overfit one batch。
2. 降低模型复杂度。
3. 检查 action normalization。
4. 使用较小 horizon 或较少 flow integration steps。

## 10. 最终目标陈述

本项目最终希望证明：

```text
Low-frequency action chunks capture most smooth robot motion,
while high-frequency corrections are most useful near transition/contact phases.
Compared with raw action chunking Flow Matching, frequency-gated action chunking
may preserve accuracy while improving smoothness and reducing boundary jerk.
```

