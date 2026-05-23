# Frequency-Gated Action Chunking Project Plan

## 1. Motivation

This project studies action representation for generative robot imitation policies. Standard action chunking predicts a sequence of future raw actions:

```text
A_t = [a_t, a_{t+1}, ..., a_{t+H-1}]
```

This representation is simple and effective, and it is commonly used by diffusion-style and flow-matching robot policies. However, manipulation trajectories usually contain multiple temporal structures:

1. Low-frequency, smooth motion patterns such as reaching, lifting, and transporting.
2. High-frequency, localized correction signals around grasping, contact, alignment, or gripper transitions.

Always predicting full-frequency raw action chunks may introduce unnecessary jitter. Predicting only smooth low-frequency actions may remove precision that is needed near contact. The central research question is:

```text
Can a generative robot policy use low-frequency action chunks for smooth motion
and selectively activate high-frequency corrections when precision is needed?
```

We will study a Frequency-Gated Action Chunking representation. The action chunk is decomposed along the temporal axis with DCT, split into low-frequency plans and high-frequency corrections, and decoded back into action space with an adaptive gate.

## 2. Project Goals

The final project compares three classes of action chunk policies:

1. Raw Action Chunking Flow Matching
   - The main baseline.
   - Directly generates future raw action chunks.

2. Low-Frequency DCT Flow Matching
   - Generates low-frequency DCT coefficients.
   - Decodes them into actions using IDCT.
   - Expected to improve smoothness, potentially at the cost of precision.

3. Frequency-Gated DCT Flow Matching
   - Generates a low-frequency plan and high-frequency corrections.
   - Learns or estimates a gate that controls how much high-frequency correction is used.
   - Targets a better trade-off between accuracy, smoothness, and boundary jerk.

The first implementation stage should complete the offline frequency analysis and the raw Flow Matching baseline. The DCT and gated policies can then reuse the same data, metrics, logging, and evaluation infrastructure.

## 3. Environment

### 3.1 Python Environment

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

### 3.2 Dataset Installation

The project uses robomimic low-dimensional imitation datasets. The recommended default is to download all simulation low-dimensional datasets:

```bash
python robomimic/robomimic/scripts/download_datasets.py \
  --tasks sim \
  --dataset_types all \
  --hdf5_types low_dim
```

This covers simulation tasks such as Lift, Can, Square, Transport, and Tool Hang. The first experiments should prioritize:

```text
Lift-PH low_dim
Can-PH low_dim
```

Lift is a reliable sanity check. Can is a better candidate for observing stronger contact and alignment effects.

If disk space is limited, start with:

```bash
python robomimic/robomimic/scripts/download_datasets.py \
  --tasks lift can \
  --dataset_types ph \
  --hdf5_types low_dim
```

### 3.3 Reproducibility Requirements

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

## 4. Code Architecture

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

### 4.1 Data Module

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

5. Normalization
   - Compute action mean/std only from the training split.
   - Compute observation mean/std only from the training split.
   - Save normalization statistics in the run directory.

### 4.2 Transform Module

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

## 5. Experimental Design

### 5.1 Experiment A: Frequency Decomposition Diagnostic

Goal: verify that action chunks contain meaningful low-frequency and high-frequency structure.

Inputs:

```text
Dataset: robomimic Lift-PH low_dim, Can-PH low_dim
Horizon H: 16
Stride: 1
K values: [2, 4, 8, 12, 16]
```

Procedure:

1. Load native actions from each demonstration.
2. Construct action chunks.
3. Apply DCT along the temporal axis.
4. For each `K`, keep only the first `K` frequency components.
5. Reconstruct low-frequency action chunks with IDCT.
6. Compute reconstruction, smoothness, high-frequency energy, and boundary jerk metrics.

Expected behavior:

1. Reconstruction MSE should decrease as `K` increases.
2. Low-frequency reconstructions should be smoother than raw actions.
3. High-frequency energy should be higher near gripper or contact-like transitions.

### 5.2 Experiment B: Gripper-Transition-Conditioned Analysis

Goal: test whether high-frequency corrections are associated with grasping or contact-like phases.

Define gripper transitions as:

```text
|a_gripper[t+1] - a_gripper[t]| > tau
```

where `tau` is the 90th or 95th percentile of gripper action differences.

Split chunks into:

```text
near transition
away from transition
```

Compare:

1. High-frequency energy ratio.
2. Oracle alpha.
3. Reconstruction MSE.
4. Smoothness.

Expected behavior:

```text
Near-transition chunks should have higher high-frequency energy
and higher oracle alpha.
```

If the effect is weak on Lift, repeat the analysis on Can.

### 5.3 Experiment C: Raw Action Chunking Flow Matching Baseline

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

Model structure:

1. Observation encoder MLP.
2. Sinusoidal time embedding.
3. Noisy action chunk encoder.
4. Velocity prediction head.

Output shape:

```text
[batch, H, action_dim]
```

Sampling:

1. Initialize an action chunk from Gaussian noise.
2. Use Euler integration from `t=0` to `t=1`.
3. Decode the normalized action chunk.
4. Unnormalize back to the original action space.

This baseline is the main reference point for all DCT and gated variants.

### 5.4 Experiment D: Low-Frequency DCT Flow Matching

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

Expected behavior:

```text
DCT low-frequency policies should be smoother than the raw baseline,
but may lose precision near transition or contact phases.
```

### 5.5 Experiment E: Frequency-Gated DCT Flow Matching

The model generates a low-frequency plan, high-frequency correction, and gate:

```text
z_low, z_high, alpha = model(o_t)
A_hat = IDCT([z_low, alpha * z_high])
```

Training can be staged:

1. Oracle gate pretraining
   - Use offline `alpha*` values as pseudo-labels.
   - Train the gate with MSE or cross entropy.

2. End-to-end decoded action training
   - Combine decoded action MSE with the Flow Matching objective.

Start with `K=8`, then tune `K in [4, 8, 12]`.

Expected behavior:

```text
The gated DCT policy should approach raw baseline accuracy,
while reducing unnecessary high-frequency jitter away from transitions.
```

## 6. Evaluation Metrics

### 6.1 Reconstruction MSE

Used for DCT diagnostics:

```text
MSE_rec(K) = mean(||A - IDCT([Z_0:K, 0])||^2)
```

Interpretation:

1. Measures how much action information is retained by low-frequency coefficients.
2. Should decrease as `K` increases.
3. If `K=4` or `K=8` is already low-error, the actions have strong low-frequency structure.

### 6.2 Smoothness

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

### 6.3 Delta Action MSE

To detect over-smoothing, evaluate delta matching:

```text
DeltaMSE(A_hat, A)
= mean(||(A_hat[:, 1:] - A_hat[:, :-1]) - (A[:, 1:] - A[:, :-1])||^2)
```

Interpretation:

1. Measures whether predicted action changes match ground-truth action changes.
2. Captures temporal dynamics better than smoothness alone.
3. Important for policy evaluation.

### 6.4 High-Frequency Energy Ratio

```text
E_high(K) = sum_{k=K}^{H-1} ||Z_k||^2 / (sum_{k=0}^{H-1} ||Z_k||^2 + eps)
```

Interpretation:

1. Measures how much chunk energy lies in high-frequency components.
2. Can be aggregated by task, phase, or transition condition.

### 6.5 Oracle Alpha

```text
alpha* = argmin_alpha [
  ||IDCT([Z_low, alpha Z_high]) - A||^2 + lambda * alpha
]
```

where:

```text
alpha in [0, 0.25, 0.5, 0.75, 1.0]
```

Interpretation:

1. `alpha*` indicates whether a chunk needs high-frequency correction.
2. Higher `alpha*` near transitions supports adaptive gating.

### 6.6 Boundary Jerk

Simulate receding-horizon execution:

```text
J_boundary = ||a_new[t] - a_old[t-1]||_2
```

Implementation:

1. Switch to a new action chunk every `K_exec` steps.
2. Compare the last executed action from the previous chunk with the first executed action from the new chunk.

Interpretation:

1. Measures discontinuity between consecutive chunks.
2. Raw action chunking may achieve low MSE but large boundary jerk.
3. Low-frequency or gated representations may reduce jerk.

### 6.7 Policy Metrics

Each policy run should report at least:

```text
val/flow_matching_loss
val/action_mse
val/smoothness
val/delta_action_mse
val/boundary_jerk
val/gripper_transition_action_mse
val/non_transition_action_mse
```

Optional rollout metrics:

```text
rollout/success_rate
rollout/return
rollout/horizon
rollout/mean_action_smoothness
```

## 7. Result Figures and Tables

The first analysis stage should generate:

1. Reconstruction MSE vs K.
2. Smoothness vs K.
3. High-frequency energy near vs away from transitions.
4. Oracle alpha near vs away from transitions.

The Flow Matching baseline stage should generate:

1. Training and validation loss curves.
2. Decoded action MSE curves.
3. Smoothness comparisons.
4. Boundary jerk comparisons.

Final comparison table:

```text
Method                    Action MSE    Smoothness    Delta MSE    Boundary Jerk
Raw FM                    ...
DCT Low-Frequency FM      ...
Gated DCT FM              ...
```

## 8. Development Order

Recommended implementation order:

1. Create config, logging, and path utilities.
2. Implement dataset inspection.
3. Implement a robust robomimic HDF5 data loader.
4. Implement action chunk construction and normalization.
5. Implement DCT / IDCT transforms.
6. Implement frequency diagnostic metrics.
7. Run Lift-PH frequency analysis.
8. Run Can-PH frequency analysis.
9. Implement the raw action chunking Flow Matching baseline.
10. Add smoothness, delta action MSE, and boundary jerk validation.
11. Implement DCT low-frequency Flow Matching.
12. Implement gated DCT Flow Matching.
13. Add rollout evaluation.

## 9. Risks and Fallbacks

### Risk 1: Lift does not show strong high-frequency effects

Fallback:

1. Use Can-PH.
2. Use boundary jerk and smoothness as primary diagnostics.
3. Treat Lift as a sanity check instead of the only evidence.

### Risk 2: Gripper transition is not a good contact proxy

Fallback:

1. Use object-state features to define task phase.
2. Use action acceleration or delta-action percentiles to define transitions.
3. Compare alignment phases in Can or Square.

### Risk 3: DCT low-frequency policy over-smooths

Fallback:

1. Add delta action MSE.
2. Use gated high-frequency correction.
3. Tune `K` and `lambda_alpha`.

### Risk 4: Flow Matching baseline is unstable

Fallback:

1. Overfit one batch first.
2. Reduce model complexity.
3. Check action normalization.
4. Use a shorter horizon or fewer flow integration steps.

## 10. Final Claim

The final project aims to test the following claim:

```text
Low-frequency action chunks capture most smooth robot motion,
while high-frequency corrections are most useful near transition/contact phases.
Compared with raw action chunking Flow Matching, frequency-gated action chunking
may preserve accuracy while improving smoothness and reducing boundary jerk.
```

