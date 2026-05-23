# Action Representation Milestone Report

Generated at: 2026-05-23 10:42:46

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

| Task | Status | Chunks | Action Dim | K=4 MSE | K=8 MSE | K=8 Smoothness | K=8 High-Energy | Metrics |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Lift | done | 6666 | 7 | 0.009255 | 0.003718 | 0.07214 | 0.01501 | outputs/analysis/frequency_diagnostic/metrics/lift_ph_frequency_diagnostic_20260522_230314.json |
| Can | done | 20207 | 7 | 0.007126 | 0.003159 | 0.05361 | 0.01236 | outputs/analysis/can_frequency_diagnostic/metrics/can_ph_frequency_diagnostic_20260522_230315.json |
| Push-T | done | 22560 | 2 | 0.000334 | 4.624e-05 | 0.002633 | 0.0004829 | outputs/analysis/pusht_frequency_diagnostic/metrics/pusht_frequency_diagnostic_20260522_230317.json |

## 6. Train / Eval Results

Currently completed `6/6` policy train/eval entries.

| Task | Method | Status | Best Epoch | Val Action MSE | Smoothness | Delta MSE | Rollouts | Success | Videos | Mean Return | Mean Steps | Metrics |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Lift | Raw UNet FM | done | 360 | 0.06343 | 0.1385 | 0.03707 | 50 | 1 | 50 | 1 | 52.04 | outputs/eval/lift_ph_fm_raw_20260522_230319/env/metrics.json |
| Lift | DCT Low-Frequency UNet FM | done | 910 | 0.06686 | 0.07163 | 0.02709 | 50 | 1 | 5 | 1 | 56.58 | outputs/eval/lift_ph_fm_dct_lowfreq_k8_20260522_231606/env/metrics.json |
| Can | Raw UNet FM | done | 920 | 0.05077 | 0.1095 | 0.03006 | 50 | 0.94 | 5 | 0.94 | 153.4 | outputs/eval/can_ph_fm_raw_20260522_232714/env/metrics.json |
| Can | DCT Low-Frequency UNet FM | done | 900 | 0.04716 | 0.05004 | 0.0214 | 50 | 0.62 | 5 | 0.62 | 240.7 | outputs/eval/can_ph_fm_dct_lowfreq_k8_20260522_235531/env/metrics.json |
| Push-T | Raw UNet FM | done | 480 | 0.02121 | 0.002452 | 0.001281 | 50 | 0.86 | 5 | 70.63 | 170.5 | outputs/eval/pusht_fm_raw_20260523_002454/env/metrics.json |
| Push-T | DCT Low-Frequency UNet FM | done | 560 | 0.02063 | 0.002258 | 0.00121 | 50 | 0.88 | 5 | 93.7 | 194.6 | outputs/eval/pusht_fm_dct_lowfreq_k8_20260523_005153/env/metrics.json |

## 7. Output Inventory

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

## 8. Current Takeaways

- Lift: raw success=1, DCT success=1; DCT smoothness change=-48.3%, action MSE change=+5.4%.
- Can: raw success=0.94, DCT success=0.62; DCT smoothness change=-54.3%, action MSE change=-7.1%.
- Push-T: raw success=0.86, DCT success=0.88; DCT smoothness change=-7.9%, action MSE change=-2.7%.
- On Lift, both raw and DCT reach 100% env-reset success, so the easy task is not discriminative enough by itself.
- On Can, DCT has better validation MSE and smoothness, but env success drops from 0.94 to 0.62. This is the strongest evidence that good low-frequency reconstruction does not guarantee closed-loop manipulation success.
- On Push-T, DCT is comparable or slightly better than raw, suggesting this planar state benchmark is friendly to low-frequency action chunks.
- The next method should train an adaptive gate or high-frequency correction: keep the low-frequency smoothness benefit on Lift/Push-T, while restoring precision on contact-sensitive Can phases.

## 9. Task Plan

The task plan is organized by evidence type, not by simply adding more benchmarks:

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
   - DROID filtered subset: offline representation diagnostics on real-robot `insert / plug / adjust / align` segments.

## 10. FAST Analysis

Reference: FAST: Efficient Action Tokenization for Vision-Language-Action Models, arXiv:2501.09747, https://arxiv.org/abs/2501.09747

FAST and the milestone Low-Frequency DCT baseline both use DCT, but they impose different bottlenecks:

1. The current Low-Frequency DCT FM baseline is a fixed low-pass model: it keeps only the first `K=8` temporal DCT coefficients and zeros out all coefficients with `k>=K`.
2. FAST is closer to compression-based tokenization: apply DCT to the full action sequence, quantize frequency-domain coefficients, and retain sparse nonzero coefficients after quantization. High-frequency coefficients can survive when their magnitude is large enough.
3. Therefore, the Can result, where fixed low-pass DCT success drops from raw 0.94 to 0.62, does not contradict FAST. It shows that a fixed low-pass bottleneck can over-smooth precision-sensitive manipulation, which is exactly where FAST-style sparse full-spectrum DCT or adaptive high-frequency correction should help.

The current results are consistent with the FAST motivation: low-frequency structure reduces action variation, but high-frequency content should not be discarded unconditionally on tasks that require grasping, alignment, and short corrections. The next comparison should evaluate fixed low-pass DCT against sparse full-spectrum DCT at matched compression ratios, measuring success, smoothness, and delta MSE.

## 11. Next Steps

Add the following methods and dataset extension to the next experiment plan:

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

All new methods should reuse as much of the current pipeline as possible: the UNet Flow Matching backbone, normalization, W&B logging, and unified `metrics.json` output. Executable benchmarks continue to save `env-reset` rollouts and simulation videos; DROID saves offline metrics, action reconstruction plots, and dataset trajectory visualizations.

## 12. Reproduction

```bash
scripts/launch_detached_milestone_pipeline.sh
tail -f logs/detached/milestone_pipeline_*.log
```

Generate reports only:

```bash
python3 scripts/generate_milestone_report.py
```
