# Action Representation for Generative Robot Policies Project Plan

## 1. Motivation

This project studies action representation for generative robot imitation policies, rather than committing to one fixed action-chunking or tokenizer formulation. The current implementation uses action chunking as the shared experimental interface: the policy predicts a sequence of future continuous actions.

```text
A_t = [a_t, a_{t+1}, ..., a_{t+H-1}]
```

The raw action chunk representation is simple and effective, and it is commonly used by diffusion-style and flow-matching robot policies. However, manipulation trajectories usually contain multiple temporal structures:

1. Low-frequency, smooth motion patterns such as reaching, lifting, and transporting.
2. High-frequency, localized correction signals around grasping, alignment, gripper command transitions, and short corrections that may be contact-relevant.

Always predicting full-frequency raw action representations may introduce unnecessary jitter. Predicting only smooth low-frequency representations may remove precision near grasping, alignment, and gripper command transitions. The central research question is:

```text
Can a generative robot policy use compact, smooth action representations
while preserving high-frequency corrections when precision is needed?
```

We start with frequency-structured action representations. An action chunk is decomposed along the temporal axis with DCT, split into a low-frequency plan and high-frequency corrections, and decoded back into action space. Adaptive frequency gating, sparse DCT, and wavelets can all be treated as candidate action representations under this broader framing. In other words, action chunking is the current train/eval interface, tokenizer is a possible future representation, and the project scope is action representation.

The first stage is no longer only an offline frequency analysis. The milestone must deliver a reproducible train/eval pipeline. Concretely, the milestone should run frequency diagnostics, a Raw UNet Flow Matching baseline, a Low-Frequency DCT UNet Flow Matching baseline, and simulation rollout evaluation on Lift-PH low_dim, the harder Can-PH low_dim task, and the Push-T state benchmark. Lift is a sanity check. Can is the primary complete robomimic task. Push-T is an additional external benchmark that tests whether the frequency-based idea transfers beyond robosuite to a planar pushing task.

## 2. Project Goals

The final project compares multiple action representations for generative robot policies. The milestone first compares three continuous action representations:

1. Raw Continuous Action Chunk Flow Matching
   - The main baseline.
   - Directly generates future raw action chunks.

2. Low-Frequency DCT Flow Matching
   - Generates low-frequency DCT coefficients.
   - Decodes them into actions using IDCT.
   - Expected to improve smoothness, potentially at the cost of precision.

3. Adaptive Frequency-Gated DCT Flow Matching
   - Generates a low-frequency plan and high-frequency corrections.
   - Learns or estimates a gate that controls how much high-frequency correction is used.
   - Targets a better trade-off between accuracy, smoothness, and boundary jerk.

Later stages can add FAST-style sparse DCT and wavelet representations. This keeps the project valid even if the final method is not a full discrete tokenizer: the core question is how action representation affects precision, smoothness, and execution stability in generative imitation policies.

The milestone stage must complete:

1. Frequency diagnostics on Lift-PH low_dim, Can-PH low_dim, and Push-T state.
2. Raw temporal-UNet Flow Matching training, checkpointing, W&B logging, and rollout evaluation.
3. Low-Frequency DCT temporal-UNet Flow Matching training, checkpointing, W&B logging, and rollout evaluation.
4. Rollout protocols:
   - `env-reset`: the official metric protocol using `env.reset()` random initial states.
   - `dataset-reset`: video/debug protocol using demonstration initial simulator states.
5. A unified metrics JSON for every checkpoint, including validation action MSE, smoothness, delta action MSE, simulation success rate, mean return, and mean steps.

Push-T uses a separate adapter: the dataset comes from the Diffusion Policy zarr replay buffer, and the environment comes from `gym-pusht`. It does not use the robomimic HDF5 loader or the robosuite rollout script. The Adaptive Gated DCT policy can remain a final-project extension. If time permits, the milestone can include oracle-gate diagnostics, but those diagnostics should not replace the complete train/eval pipeline.

DROID is added as a final-project scale-up dataset. DROID is an in-the-wild robot manipulation dataset; the official paper/site report about 76k demonstration trajectories and 350 hours of interaction data across hundreds of real scenes and dozens of tasks. Its role is not to replace robomimic / Push-T closed-loop simulation evaluation. Instead, DROID tests whether the frequency tokenizer scales to real-robot, multi-scene, multi-task action data. The DROID stage should first run offline frequency/tokenizer analysis and tokenizer / gate pretraining. Unless a real-robot or reproducible evaluator is connected, it should not report simulation success.

## 3. Method Principle

### 3.1 Action Chunk Representation

The raw action at a single timestep is:

```text
a_t in R^d
```

where `d` is the action dimension. The default action representation uses:

```text
rel_pos: 3D
rel_rot_6d: 6D
gripper: 1D
```

Therefore:

```text
d = 10
```

An action chunk is a sequence of future actions starting at time `t`:

```text
A_t = [a_t, a_{t+1}, ..., a_{t+H-1}]
```

If the horizon is `H=16`, then:

```text
A_t shape = [H, d] = [16, 10]
```

The first dimension is time, and the second dimension is action channel.

### 3.2 Why DCT Keeps the Shape `[H, d]`

This project applies DCT along the temporal dimension of each action chunk:

```text
Z_t = DCT(A_t, axis=time)
```

DCT does not change the tensor shape. It changes the meaning of the temporal axis from time index to frequency index. Therefore:

```text
A_t shape = [H, d]
Z_t shape = [H, d]
```

The meaning changes as follows:

```text
A_t[i, j] = action value at future timestep i and action dimension j
Z_t[k, j] = DCT coefficient for action dimension j at temporal frequency k
```

For each action dimension, DCT maps a length-`H` time sequence to a length-`H` sequence of frequency coefficients. Low-index DCT coefficients represent slow trends; high-index coefficients represent fast changes or localized corrections.

### 3.3 Meaning of IDCT

IDCT stands for inverse DCT. DCT maps an action sequence from the time domain into the frequency domain, while IDCT maps frequency coefficients back into a time-domain action sequence:

```text
Z_t = DCT(A_t)
A_t = IDCT(Z_t)
```

With orthonormal DCT normalization, DCT and IDCT are theoretically lossless inverses. Therefore, if all frequency coefficients are retained:

```text
Z_full = Z_t
A_recon = IDCT(Z_full)
```

then:

```text
A_recon ≈ A_t
```

where the only error is numerical precision.

In this project, IDCT does not change the action semantics. Its role is to decode a modified frequency representation back into the original action space, so we can compute MSE, smoothness, and eventually execute the predicted actions. For example:

```text
raw action chunk A_t
  -> DCT gives frequency coefficients Z_t
  -> remove or scale selected high-frequency coefficients
  -> IDCT gives a new action chunk A_hat
```

Both low-frequency reconstruction and adaptive gating rely on IDCT:

```text
A_low = IDCT([Z_low, 0])
A_gated = IDCT([Z_low, alpha * Z_high])
```

Thus, DCT/IDCT is an action representation transform: the model or analysis can operate in frequency space, while evaluation and execution remain in time-domain action chunk space.

### 3.4 Meaning of `K`

`K` is the number of low-frequency DCT coefficients retained:

```text
Z_low = Z_t[0:K]
Z_high = Z_t[K:H]
```

For example, with `H=16`:

```text
K=2: keep only the coarsest low-frequency trend
K=4: keep a small number of low-frequency components
K=8: keep half of the frequencies, often a smoothness-accuracy compromise
K=12: close to the full action chunk
K=16: equal to H, no frequency truncation, exact reconstruction in theory
```

Low-frequency reconstruction zeroes out the high-frequency coefficients:

```text
Z_hat[0:K] = Z_t[0:K]
Z_hat[K:H] = 0
A_low = IDCT(Z_hat)
```

If `A_low` has low MSE against the original `A_t`, then the chunk is mostly explained by low-frequency structure. If the MSE is high, high-frequency coefficients carry important information.

### 3.5 High-Frequency Residual and Adaptive Gate

The full action chunk can be written as a low-frequency plan plus high-frequency correction:

```text
A_t = IDCT([Z_low, Z_high])
```

The adaptive gate uses high-frequency correction only when needed:

```text
A_hat = IDCT([Z_low, alpha * Z_high])
```

where:

```text
alpha in [0, 1]
```

Intuition:

```text
alpha = 0: use only the low-frequency plan, producing smoother actions
alpha = 1: use the full high-frequency correction, improving precision but potentially adding jitter
0 < alpha < 1: trade off smoothness and precision
```

In the final policy, `alpha` can be predicted from the current observation:

```text
alpha = g(o_t)
```

The first stage does not treat `alpha*` as a true label. It is a diagnostic quantity: if chunks near gripper command transitions require more high-frequency correction across multiple `lambda` values, adaptive gating is better motivated.

### 3.6 Relation to the Flow Matching Baseline

The Raw Action Chunking Flow Matching baseline directly generates the raw action chunk:

```text
model(o_t) -> A_t
```

The DCT version generates a frequency-domain representation:

```text
model(o_t) -> Z_t
A_t = IDCT(Z_t)
```

The adaptive gated version further decomposes the frequency-domain representation into low frequency, high frequency, and a gate:

```text
model(o_t) -> Z_low, Z_high, alpha
A_hat = IDCT([Z_low, alpha * Z_high])
```

Thus, all three methods share the same data loader, normalization, evaluation metrics, and receding-horizon evaluation. They differ only in the action chunk generation space.

## 4. Environment

### 4.1 Python Environment

Use the existing `cs224r` conda environment:

```bash
conda activate cs224r
```

Install local robosuite and robomimic:

```bash
python -m pip install -e robosuite
python -m pip install -e robomimic
```

Install project-level dependencies:

```bash
python -m pip install numpy h5py scipy matplotlib pandas pyyaml omegaconf tqdm
```

Flow Matching training requires PyTorch. Robomimic already includes `torch` in its dependency list, but it is useful to verify the installation:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

### 4.2 Dataset Installation

The project uses robomimic low-dimensional imitation datasets. The recommended default is to download all simulation low-dimensional datasets:

```bash
python robomimic/robomimic/scripts/download_datasets.py \
  --tasks sim \
  --dataset_types all \
  --hdf5_types low_dim
```

This covers simulation tasks such as Lift, Can, Square, Transport, and Tool Hang. The first robomimic experiments should prioritize:

```text
Lift-PH low_dim
Can-PH low_dim
```

Lift is a reliable sanity check. Can is a better candidate for observing stronger grasping, alignment, and contact-relevant effects. The first stage does not directly observe physical contact; it uses gripper command transitions as a proxy for contact-relevant phases.

Push-T requires additional dependencies and data:

```bash
scripts/setup_pusht_deps.sh
python scripts/download_pusht_dataset.py
```

The default Push-T dataset path is:

```text
data/pusht/pusht_cchi_v7_replay.zarr
```

Push-T actions are 2D target positions `[x, y]`, and the default observation is a 5D state `[agent_x, agent_y, block_x, block_y, block_angle]`. Push-T has no gripper transition, so frequency analysis should focus on phase, action smoothness, and rollout success rather than a gripper proxy.

DROID is the planned scale-up dataset:

```text
Official page: https://droid-dataset.github.io/
Paper: https://arxiv.org/abs/2403.12945
Planned local path: data/droid/
```

DROID should not use the robomimic HDF5 loader, and it does not provide a local simulator for direct `env-reset` evaluation. Add a `DROIDChunkDataset` or RLDS / HuggingFace adapter that maps real-robot trajectories into the shared action chunk interface:

```text
observation: image / proprio / language metadata; first stage can use proprio-only or cached visual features
action: canonical end-effector delta pose + gripper
rotation: convert to rot_6d or another continuous representation
chunk: [H, d], using the same DCT / tokenizer interface as robomimic and Push-T
```

The first DROID experiments should use offline metrics only: reconstruction MSE, smoothness, delta MSE, compression ratio, high-frequency energy ratio, per-action-group spectra, and dataset-video qualitative visualization. DROID should not be included in the milestone simulation success table; it is for final-project real-world scale-up, tokenizer pretraining, and cross-task frequency-structure validation.

Candidate datasets for precision insertion, dexterous fingers, and small contact adjustment:

```text
ManiSkill: https://maniskill.readthedocs.io/
D4RL / Adroit: https://github.com/Farama-Foundation/d4rl/wiki/Tasks
Gymnasium-Robotics Adroit: https://robotics.farama.org/main/envs/adroit_hand/adroit_door/
RoboHive datasets: https://github.com/vikashplus/robohive/wiki/7.-Datasets
Meta-World tasks: https://metaworld.farama.org/benchmark/task_descriptions/
```

Recommended priority:

1. Precision insertion / plug-in: use ManiSkill first.
   - `PegInsertionSide-v1`: peg-in-hole side insertion, directly matching precision insertion and alignment.
   - `PlugCharger-v1`: pick up a charger and insert it into a receptacle, directly matching plug / socket behavior.
   - ManiSkill supports state / RGBD observations, downloadable demonstrations, and executable simulation rollouts, so it is the best next benchmark family for the current train/eval pipeline.

2. Current robomimic supplements: Square and Tool Hang.
   - `Square`: nut assembly that requires grasping, alignment, and placement onto a peg; it is a precision placement / assembly task.
   - `Tool Hang`: a longer-horizon task with fine alignment and contact-adjustment behavior.
   - These are compatible with the existing robomimic loader and have the lowest integration cost, but they are not true insertion or dexterous-hand tasks.

3. Dexterous fingers: use D4RL / Adroit or RoboHive first.
   - Adroit/D4RL includes `pen`, `door`, `hammer`, and `relocate` datasets for 24-DoF Shadow Hand manipulation.
   - RoboHive also provides dexterous manipulation / Shadow Hand environments and datasets.
   - These tasks have different action dimensions and control semantics from the Panda gripper tasks. They need a separate `DexterousHandChunkDataset` adapter and should not be mixed into the current 7-D / 10-D end-effector action pipeline.

4. Small contact adjustment: use ManiSkill plus filtered DROID.
   - ManiSkill insertion / plug / assembly / poke / pull tasks provide executable simulation evaluators.
   - DROID can be filtered with language annotations such as `insert`, `plug`, `place into`, `adjust`, `align`, `push slightly`, and `move a little`, combined with gripper close, end-effector speed / acceleration, and small action deltas to locate small contact-adjustment segments.
   - DROID filtering must start with data auditing: report episode count, action distribution, and video examples. Do not assume every keyword-filtered segment corresponds to physical contact.

If disk space is limited, start with:

```bash
python robomimic/robomimic/scripts/download_datasets.py \
  --tasks lift can \
  --dataset_types ph \
  --hdf5_types low_dim
```

### 4.3 Reproducibility Requirements

Each run should save:

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

All paths, datasets, model options, training parameters, and evaluation parameters should be config-driven.

## 5. Code Architecture

This project should use a final-project architecture rather than a temporary milestone layout:

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

### 5.1 Data Module

The data layer reads robomimic HDF5 demonstrations and constructs action chunks.

Requirements:

1. HDF5 lazy loading
   - Avoid sharing or pickling open HDF5 handles across dataloader workers.

2. No cross-episode chunks
   - Build chunk indices separately for each demonstration.

3. Demo-level train/validation split
   - Avoid leakage between adjacent chunks from the same trajectory.

4. Config-driven observation keys
   - Low-dimensional observations may include keys such as `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, and `object`.
   - The selected observation keys should be controlled by config.

5. Action representation
   - The preferred action representation is the concatenation of `action_dict/rel_pos`, `action_dict/rel_rot_6d`, and `action_dict/gripper`, giving a 10D action.
   - 6D rotation is preferred to avoid discontinuities in axis-angle actions that can create spurious high-frequency energy.
   - If `action_dict` is missing, run robomimic's `extract_action_dict.py` first. Legacy `actions` should only be used as a sanity-check fallback.

6. Normalization
   - The default action normalization is per-dimension min-max normalization to `[-1, 1]`.
   - Min/max statistics are computed only from the training split to avoid validation leakage.
   - Observation normalization uses per-dimension z-score statistics from the training split.
   - Frequency energy and MSE are primarily computed in normalized action space; raw action-space metrics can be included as supplementary results.
   - Action z-score normalization can be used as an ablation, but it is not the default.

### 5.2 Transform Module

`transforms/dct.py` should provide:

```python
dct_time(x)
idct_time(z)
```

Input shape:

```text
[batch, horizon, action_dim]
```

DCT is applied along the horizon dimension.

`frequency_split.py` should provide:

```python
split_low_high(z, k)
low_reconstruct(actions, k)
high_energy_ratio(z, k)
```

`gates.py` should provide:

```python
oracle_alpha(actions, z, k, alphas, lambda_alpha)
```

The gated model should reuse these utilities.

## 6. Experimental Design

### 6.1 Experiment A: Frequency Decomposition Diagnostic

Goal: verify that action chunks contain meaningful low-frequency and high-frequency structure.

Inputs:

```text
Dataset: robomimic Lift-PH low_dim, robomimic Can-PH low_dim, Push-T state
Horizon H: 16
Stride: 1
K values: [2, 4, 8, 12, 16]
```

The DROID extension uses the same Experiment A offline diagnostic, but it does not enter the simulation rollout table:

```text
Dataset: DROID subset/full split
Horizon H: 16 or dataset-control-rate-adjusted horizon
Stride: 1 for coverage, stride=H for non-overlapping sanity check
Metrics: reconstruction MSE, smoothness, delta MSE, compression ratio, high-frequency energy
```

Notes:

1. `K=16` equals `K=H`, so it is the trivial full-reconstruction endpoint with zero or near-zero numerical MSE.
2. `stride=1` covers all possible chunks, but adjacent chunks strongly overlap and should not be treated as independent samples for overconfident error bars.
3. Aggregate statistics should report mean / median by default; confidence intervals should use demo-level bootstrap.
4. Also run a `stride=H` non-overlapping sanity check to verify that conclusions are not caused by chunk overlap.

Procedure:

1. Load native actions from each demonstration.
2. Construct action chunks.
3. Apply DCT along the temporal axis.
4. For each `K`, keep only the first `K` frequency components.
5. Reconstruct low-frequency action chunks with IDCT.
6. Compute reconstruction, smoothness, and high-frequency energy metrics.
7. Report both aggregate spectra and per-action-group spectra.

Expected behavior:

1. Reconstruction MSE should decrease as `K` increases.
2. Low-frequency reconstructions should be smoother than raw actions.
3. High-frequency energy should be higher near gripper command transitions.

Per-action-group analysis should at least separate:

```text
translation: rel_pos
rotation: rel_rot_6d
gripper: gripper
```

This separates high-frequency energy that comes only from the gripper channel from high-frequency structure that also affects arm motion.

### 6.2 Experiment B: Gripper-Transition-Conditioned Analysis

Goal: test whether high-frequency corrections are associated with gripper command transitions. A gripper transition is only a proxy for contact-relevant phases; it is not the same as physical contact.

Define gripper transitions as:

```text
|a_gripper[t+1] - a_gripper[t]| > tau
```

Threshold selection:

1. First inspect the gripper action distribution with `inspect_dataset.py`.
2. If gripper actions are approximately binary, use `|delta_gripper| > 1.0` by default.
3. If they are not binary, use the 90th or 95th percentile as a fallback.

Split chunks into:

```text
near transition
away from transition
```

Compare:

1. High-frequency energy ratio.
2. Oracle alpha and relative MSE reduction.
3. Reconstruction MSE.
4. Smoothness.

Expected behavior:

```text
Near-transition chunks should have higher high-frequency energy,
higher relative MSE reduction from high-frequency coefficients,
and possibly higher oracle alpha under a range of lambda values.
```

If the effect is weak on Lift, repeat the analysis on Can. Push-T has no gripper channel, so it should not use the gripper-transition-conditioned split; instead, use task phase / progress-conditioned smoothness and rollout success as the complementary analysis.

On DROID, transition analysis must explicitly distinguish gripper command transitions from physical contact. The first version should use gripper command, end-effector speed / acceleration, action-delta percentiles, and language/task metadata as phase proxies. Unless reliable force/contact annotations are available, these proxies should not be described as physical contact.

Experiment A plan for the new task families:

1. ManiSkill precision insertion / plug-in
   - Use `PegInsertionSide-v1` and `PlugCharger-v1` first.
   - Run the same DCT / sparse DCT / wavelet offline diagnostics.
   - Since these tasks have executable simulation environments, later add Flow Matching train/eval and rollout success.

2. robomimic Square / Tool Hang
   - Use them as low-cost extensions of the current robomimic pipeline.
   - Focus on whether low-pass DCT fails during precision assembly phases, and whether FAST-style sparse DCT restores success.

3. D4RL / Adroit or RoboHive dexterous hand
   - Start with offline chunk frequency/tokenizer diagnostics only.
   - Add hand-action normalization, per-joint spectra, finger-group smoothness, and high-dimensional action chunk reconstruction.
   - Add closed-loop evaluation only after the hand environment adapter is stable.

4. DROID contact-adjustment filtered subset
   - Generate candidate subsets using language keywords and trajectory statistics.
   - Manually inspect videos to verify insertion, alignment, and small contact-adjustment behavior.
   - Report subset size, keyword distribution, action magnitude distribution, and high-frequency energy distribution.

### 6.2.1 Task Plan for Evaluating Reconstruction Smoothness

To show that reconstruction smoothness is not only an offline metric improvement but also useful for control, task selection should include positive evidence, sanity checks, and negative trade-off evidence:

1. Positive evidence: Push-T
   - Push-T is the cleanest current smoothness task: continuous planar pushing benefits from stable, low-jitter actions.
   - In the current results, the DCT representation reduces action variation while maintaining or improving success / return, so it can support the claim that smoother reconstructed actions can help or at least not hurt smooth control.
   - Later add harder Push-T variants, such as tighter goals, longer horizons, harder initial states, or obstacles.

2. Sanity evidence: Lift
   - Lift is mostly reaching and lifting, so it is naturally low-frequency.
   - DCT and raw both succeed, showing that smooth reconstruction does not break simple manipulation.
   - Lift is too easy to prove smoothness usefulness by itself.

3. Trade-off / negative evidence: Can
   - On Can, DCT has better smoothness and validation MSE but much worse closed-loop success.
   - This task shows that smoothness alone is insufficient and precision-sensitive phases require high-frequency information.
   - It is the main motivation for sparse DCT, adaptive gating, and wavelets.

4. Future stress tests
   - ManiSkill `PegInsertionSide-v1` / `PlugCharger-v1`: test the smoothness-precision conflict in insertion, plug-in, and small contact adjustment.
   - robomimic Square / Tool Hang: low-cost tests for whether precision assembly phases require high-frequency corrections.
   - DROID filtered subsets: offline tests for whether real-robot `insert / plug / adjust / align` segments need high-frequency retention more than fixed low-frequency reconstruction.

Reports should clearly separate:

```text
Push-T: smoothness-positive task
Lift: smoothness sanity-check task
Can / insertion / contact-adjustment: smoothness-precision trade-off task
```

### 6.3 Experiment C: Raw Action Chunking Flow Matching Baseline

This is the standard action chunking baseline.

Target:

```text
condition: current observation o_t
target: raw action chunk A_t = [a_t, ..., a_{t+H-1}]
```

Flow Matching training:

```text
x0 ~ N(0, I)
x1 = normalized raw action chunk
t ~ Uniform(0, 1)
xt = (1 - t) x0 + t x1
target velocity = x1 - x0
model output = v_theta(xt, t, o_t)
loss = MSE(v_theta, x1 - x0)
```

The model uses a temporal UNet rather than a simple MLP:

1. Observation history encoder: default `observation_horizon=2`, conditioning on the two most recent low-dimensional observations.
2. Sinusoidal Flow Matching time embedding.
3. Temporal 1D UNet: input and output are both action sequences shaped `[batch, H, action_dim]`.
4. EMA weights: evaluation and checkpointing use EMA parameters by default.

Output shape:

```text
[batch, H, action_dim]
```

Sampling:

1. Initialize an action chunk from Gaussian noise.
2. Use Euler integration from `t=0` to `t=1`.
3. Decode the normalized action chunk.
4. Unnormalize back to the original action space.

This baseline is the main reference point for all DCT and gated variants. In the milestone, it must be trained and evaluated on both Lift and Can.

### 6.4 Experiment D: Low-Frequency DCT Flow Matching

The target changes from raw action chunks to low-frequency DCT coefficients.

Training target:

```text
z = DCT(A)
target = z[:K]
```

After generating low-frequency coefficients, decode with zero-padded DCT coefficients:

```text
A_hat = IDCT([z_low, 0])
```

Evaluation is still performed in action space:

1. Decoded action MSE.
2. Smoothness.
3. Delta action MSE.
4. Boundary jerk.

In the milestone, the DCT low-frequency FM and Raw FM baselines should share the same temporal UNet, observation horizon, normalization, W&B logging, and rollout protocols. Only the training target space changes.

Expected behavior:

```text
DCT low-frequency policies should be smoother than the raw baseline,
but may lose precision near gripper command transitions or contact-relevant phases.
```

### 6.5 Experiment E: Adaptive Frequency-Gated DCT Flow Matching

The model generates a low-frequency plan, high-frequency correction, and gate:

```text
z_low, z_high, alpha = model(o_t)
A_hat = IDCT([z_low, alpha * z_high])
```

Training can be staged:

1. Oracle gate pretraining
   - Use offline `alpha*` values as pseudo-labels.
   - `alpha*` is not a ground-truth gate; it is a lambda-sensitive diagnostic quantity and optional warm-start target.
   - Train the gate with MSE or cross entropy.

2. End-to-end decoded action training
   - Combine decoded action MSE with the Flow Matching objective.

Report `K=4` and `K=8` in the diagnostic stage. Start training with `K=8`, then tune `K in [4, 8, 12]`.

Expected behavior:

```text
The adaptive gated DCT policy should approach raw baseline accuracy,
while reducing unnecessary high-frequency jitter away from transitions.
```

### 6.6 FAST Comparison and Next Tokenizer Directions

FAST (Frequency-space Action Sequence Tokenization) also uses DCT, but it is not the same method as the current milestone Low-Frequency DCT baseline. The current baseline keeps only the first `K=8` low-frequency coefficients and zeros out every higher-frequency coefficient; this is a hard low-pass bottleneck. FAST is closer to time-series compression: apply DCT to the full action sequence, quantize frequency-domain coefficients, and keep sparse nonzero coefficients after quantization. In other words, FAST does not simply discard high-frequency components. It can retain high-frequency coefficients when their magnitude is large enough to survive compression. If gripper transitions, alignment corrections, or contact-relevant corrections create large high-frequency DCT coefficients, a FAST-style tokenizer can preserve them.

This distinction is important for interpreting the current results. On Can, Low-Frequency DCT obtains better validation MSE and smoothness but much worse closed-loop success. This does not contradict FAST. It suggests that a fixed low-pass DCT bottleneck over-smooths precision-sensitive manipulation, motivating sparse high-frequency retention or adaptive high-frequency correction.

The next stage should prioritize two tokenizer-inspired / representation directions:

- Method 1: FAST-style Sparse DCT Tokenizer
   - Quantize and sparsely retain full DCT coefficients instead of truncating to `K=8`.
   - First implement a continuous version: retain top-magnitude DCT coefficients or thresholded nonzero coefficients, then decode with IDCT.
   - Then consider a discrete version: quantization plus BPE / vocabulary compression for closer alignment with VLA-style tokenization.
   - Main question: at the same compression ratio, does sparse full-spectrum DCT preserve Can success better than fixed low-pass DCT while keeping the smoothness benefit?

- Method 4: Wavelet Tokenizer
   - Replace pure DCT with a multi-resolution temporal basis.
   - Wavelets are better matched to localized jumps, so they may fit gripper transitions and short corrections better than fixed low-pass DCT.
   - Main question: can wavelets preserve local high-frequency events on precision-sensitive Can phases while maintaining smooth reaching / transporting behavior?

Both directions should reuse the existing train/eval pipeline: the same temporal UNet Flow Matching backbone, normalization, `env-reset` rollout, metrics JSON, and simulation-video saving logic. This keeps new representation results directly comparable to Raw FM and Low-Frequency DCT FM.

### 6.7 DROID Real-World Dataset Scale-Up

DROID is added as the real-world scale-up stage after the current robomimic and Push-T pipeline is stable. The immediate goal is not closed-loop rollout success, because DROID is a real-world dataset without a local simulator reset protocol. Instead, DROID should answer whether the tokenizer design learned from Lift / Can / Push-T still makes sense on diverse in-the-wild robot trajectories.

Planned DROID usage:

1. Build a DROID action-chunk adapter.
   - Load RLDS / HuggingFace / local DROID trajectories.
   - Map actions into a canonical representation such as end-effector delta pose plus gripper.
   - Convert rotations to `rot_6d` or another continuous representation to avoid discontinuities.
   - Resample or window trajectories into `[H, d]` chunks.

2. Run offline tokenizer diagnostics.
   - Fixed low-pass DCT reconstruction.
   - FAST-style sparse DCT reconstruction at matched compression ratios.
   - Wavelet reconstruction around localized action jumps.

3. Use DROID for pretraining.
   - Pretrain sparse/wavelet representations or adaptive gates on diverse real-world trajectories.
   - Fine-tune / evaluate policy behavior on robomimic Can, Push-T, or another executable benchmark.
   - Report DROID as offline pretraining and representation evidence, not as simulation success.

4. Save qualitative artifacts.
   - Dataset trajectory visualizations.
   - Reconstructed action plots.
   - Optional videos comparing original action replay metadata with reconstructed/tokenized actions.

Primary DROID metrics:

```text
reconstruction MSE
delta action MSE
smoothness
compression ratio
high-frequency energy ratio
per-action-group spectra
transition-proxy-conditioned reconstruction error
```

If a real-robot evaluation path becomes available, DROID-pretrained tokenizer / policy can be evaluated with task success. Until then, DROID remains an offline scale-up and pretraining dataset in this plan.

## 7. Evaluation Metrics

### 7.0 Rollout Protocol

The milestone report must distinguish two simulation evaluation protocols:

1. `env-reset` official evaluation
   - Uses `env.reset()` with random environment initial states.
   - Run at least `50` rollouts per method and task; `10` rollouts are acceptable for debugging.
   - This is the official protocol for success rate, mean return, and mean steps.

2. `dataset-reset` video/debug evaluation
   - Resets to the initial simulator state from dataset demonstrations.
   - Used for comparable qualitative videos and policy debugging.
   - It should not be the only evidence for final success rate because it is closer to the training distribution than random env reset.

Every evaluation should save:

```text
simulation_rollout.json
metrics.json
simulation_rollout_*.mp4
```

The `metrics.json` should include validation action MSE, smoothness, delta action MSE, success rate, mean return, and mean steps.

### 7.1 Reconstruction MSE

Used for DCT diagnostics:

```text
MSE_rec(K) = mean(||A - IDCT([Z_0:K, 0])||^2)
```

Interpretation:

1. Measures how much action information is retained by low-frequency coefficients.
2. Should decrease as `K` increases.
3. If `K=4` or `K=8` is already low-error, the actions have strong low-frequency structure.

### 7.2 Smoothness

Core smoothness metric:

```text
S(A) = 1 / (H - 1) * sum_{i=0}^{H-2} ||a_{i+1} - a_i||_2^2
```

Interpretation:

1. Measures temporal jitter inside an action chunk.
2. Lower values indicate smoother actions.
3. Compute it for raw actions, DCT reconstructions, and policy predictions.

Important caveat:

```text
Smoothness should not be evaluated alone.
A nearly constant action sequence can be smooth but inaccurate.
```

Therefore, smoothness must be interpreted together with action MSE.

### 7.3 Delta Action MSE

To detect over-smoothing, evaluate delta matching:

```text
DeltaMSE(A_hat, A)
= mean(||(A_hat[:, 1:] - A_hat[:, :-1]) - (A[:, 1:] - A[:, :-1])||^2)
```

Interpretation:

1. Measures whether predicted action changes match ground-truth action changes.
2. Captures temporal dynamics better than smoothness alone.
3. Important for policy evaluation.

### 7.4 High-Frequency Energy Ratio

```text
E_high(K) = sum_{k=K}^{H-1} ||Z_k||^2 / (sum_{k=0}^{H-1} ||Z_k||^2 + eps)
```

Interpretation:

1. Measures how much chunk energy lies in high-frequency components.
2. Can be aggregated by task, phase, or transition condition.
3. Must be reported both as an aggregate ratio and per action group, so the gripper dimension does not dominate all conclusions.

### 7.5 Oracle Alpha

```text
alpha* = argmin_alpha [
  ||IDCT([Z_low, alpha Z_high]) - A||^2 + lambda * alpha
]
```

where:

```text
alpha in [0, 0.25, 0.5, 0.75, 1.0]
lambda in [1e-4, 1e-3, 1e-2, 1e-1]
```

Interpretation:

1. When `alpha=1`, DCT/IDCT fully reconstructs the original chunk, so reconstruction MSE is zero or near numerical error.
2. Therefore `alpha*` is not a ground-truth gate. It is a diagnostic quantity determined by high-frequency reconstruction gain and the `lambda` penalty.
3. Sweep `lambda` and report how the `alpha*` distribution changes, instead of choosing a single favorable penalty.
4. Report oracle analysis by default for `K=4` and `K=8`.
5. Higher `alpha*` near transitions across multiple lambda values supports adaptive gating.

Also report the more robust relative MSE reduction:

```text
RelGain(alpha)
= (MSE_low - MSE_alpha) / (MSE_low + eps)
```

Here `MSE_low` is the low-frequency reconstruction error with `alpha=0`, and `MSE_alpha` is the reconstruction error after using a given `alpha`. This metric is less sensitive to the choice of `lambda` than a single `alpha*`.

### 7.6 Boundary Jerk

Simulate receding-horizon execution:

```text
J_boundary = ||a_new[t] - a_old[t-1]||_2
```

Implementation:

1. `K_exec` is the number of actions executed from a predicted chunk before re-predicting; it is the analog of Diffusion Policy's action horizon `T_a`.
2. Switch to a new action chunk every `K_exec` steps.
3. Compare the last executed action from the previous chunk with the first executed action from the new chunk.

Interpretation:

1. Measures discontinuity between consecutive chunks.
2. Raw action chunking may achieve low MSE but large boundary jerk.
3. Low-frequency or gated representations may reduce jerk.
4. In the offline frequency diagnostic, boundary jerk is only an auxiliary metric on reconstructed chunks. It becomes a primary evaluation metric after training receding-horizon policies.

### 7.7 Policy Metrics

Each policy run should report at least:

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

Optional rollout metrics:

```text
rollout/success_rate
rollout/return
rollout/horizon
rollout/mean_action_smoothness
```

## 8. Result Figures and Tables

The first analysis stage should generate:

1. Reconstruction MSE vs K.
2. Smoothness vs K.
3. High-frequency energy near vs away from transitions.
4. Per-action-group high-frequency spectra.
5. Relative MSE reduction near vs away from transitions.
6. Oracle alpha near vs away from transitions under a lambda sweep.

The Flow Matching baseline stage should generate:

1. Training and validation loss curves.
2. Decoded action MSE curves.
3. Smoothness comparisons.
4. Boundary jerk comparisons.
5. `env-reset` simulation success rate.
6. `dataset-reset` qualitative videos.

Final comparison table:

```text
Task    Method                    Action MSE    Smoothness    Delta MSE    Env Success    Mean Steps
Lift    Raw UNet FM               ...
Lift    DCT Low-Frequency UNet FM ...
Can     Raw UNet FM               ...
Can     DCT Low-Frequency UNet FM ...
```

## 9. Development Order

Recommended implementation order:

1. Create config, logging, and path utilities.
2. Implement dataset inspection.
3. Implement a robust robomimic HDF5 data loader.
4. Implement action chunk construction and normalization.
5. Implement DCT / IDCT transforms.
6. Implement frequency diagnostic metrics.
7. Inspect the gripper action distribution and choose the transition threshold.
8. Run Lift-PH frequency analysis.
9. Run Can-PH frequency analysis.
10. Implement demo-level bootstrap and a `stride=H` sanity check.
11. Implement the temporal-UNet raw action chunking Flow Matching baseline.
12. Add smoothness, delta action MSE, and boundary jerk validation.
13. Implement DCT low-frequency temporal-UNet Flow Matching.
14. Add `env-reset` official rollout evaluation.
15. Add `dataset-reset` video/debug evaluation.
16. Run full train/eval on Lift as a sanity check.
17. Run full train/eval on Can as the primary robomimic task.
18. Run full train/eval on Push-T state as the external pushing benchmark.
19. Aggregate unified metrics JSON files and comparison tables.
20. Implement adaptive gated DCT Flow Matching.
21. Implement a FAST-style sparse DCT tokenizer baseline and compare fixed low-pass DCT against sparse full-spectrum DCT.
22. Implement a wavelet representation baseline and test fidelity around localized high-frequency events.
23. Implement a DROID adapter and first run DROID offline frequency/tokenizer diagnostics.
24. Use DROID for sparse/wavelet representation or adaptive-gate pretraining, then fine-tune / evaluate on an executable benchmark.
25. Add ManiSkill `PegInsertionSide-v1` / `PlugCharger-v1` as executable precision insertion and plug-in benchmarks.
26. Add robomimic Square / Tool Hang as low-cost precision assembly extensions.
27. Add D4RL / Adroit or RoboHive dexterous hand data, starting with offline hand-action representation diagnostics.
28. Filter DROID language annotations for insertion / alignment / small contact-adjustment subsets, then manually audit videos.

## 10. Risks and Fallbacks

### Risk 1: Lift does not show strong high-frequency effects

Fallback:

1. Use Can-PH.
2. Use per-action-group spectra, relative MSE reduction, and smoothness as primary diagnostics.
3. Treat Lift as a sanity check instead of the only evidence.

### Risk 2: Gripper transition is not a good contact proxy

Fallback:

1. Use object-state features to define task phase.
2. Use action acceleration or delta-action percentiles to define transitions.
3. Compare alignment phases in Can or Square.
4. In the final claim, describe current evidence as gripper command transition evidence, not direct physical contact evidence.

### Risk 3: DCT low-frequency policy over-smooths

Fallback:

1. Add delta action MSE.
2. Use gated high-frequency correction.
3. Tune `K` and `lambda_alpha`.

### Risk 4: Oracle alpha is too sensitive to lambda

Fallback:

1. Sweep `lambda in [1e-4, 1e-3, 1e-2, 1e-1]`.
2. Report relative MSE reduction as the main diagnostic.
3. Treat `alpha*` as an optional gate warm-start target, not a ground-truth label.

### Risk 5: Flow Matching baseline is unstable

Fallback:

1. Overfit one batch first.
2. Reduce model complexity.
3. Check action normalization.
4. Use a shorter horizon or fewer flow integration steps.

## 11. Final Claim

The final project aims to test the following claim:

```text
Compact action representations can capture most smooth robot motion,
but precision-sensitive phases require preserving selected high-frequency structure.
Compared with raw continuous action chunks and fixed low-frequency bottlenecks,
adaptive or sparse action representations may preserve accuracy while improving smoothness
and reducing boundary jerk.
```
