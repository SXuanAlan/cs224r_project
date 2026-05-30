# Next-Stage Experiment Results

Date: 2026-05-29

All new simulation numbers below use 50 env-reset rollouts with H=16 and K_exec=8.

## Updated 3x3 Results

| Task | Raw FM | DCT Low-Freq K=8 | DCT Sparse Full-Spectrum K=8 |
|---|---:|---:|---:|
| Lift | 1.00 | 1.00 | 1.00 |
| Can | 0.94 | 0.62 | 0.953 +/- 0.042 |
| Push-T | 0.86 | 0.88 | 0.80 |

Can sparse K=8 multi-seed validation summary: success `0.953 +/- 0.042`, val action MSE `0.0506 +/- 0.0037`, smoothness `0.0670 +/- 0.0019`, delta action MSE `0.0240 +/- 0.0010`. The per-seed successes were `0.92`, `1.00`, and `0.94`, so the original high Can sparse result is not a one-seed fluke. The Push-T sparse/full-spectrum cell uses the existing Push-T full-spectrum result at `outputs/eval/pusht_fm_dct_fullfreq_20260529_163603/env/metrics.json`.

Key output files:
- P1a: `outputs/analysis/p1a_can_sparse_seeds/metrics.json`, `outputs/analysis/p1a_can_sparse_seeds/figures/seed_variance.png`
- P2: `outputs/eval/lift_fm_dct_sparse_k8_20260529_174947/env/metrics.json`
- P3: `outputs/analysis/p3_can_k_sweep/metrics/k_sweep.json`

## P1.b Can Per-Phase DCT Spectrum

![Can phase spectrum aggregate](../../outputs/analysis/p1b_can_phase_spectrum/figures/can_phase_spectrum_aggregate.png)

![Can phase spectrum per channel](../../outputs/analysis/p1b_can_phase_spectrum/figures/can_phase_spectrum_per_channel.png)

Transition chunks have substantially more normalized high-frequency energy past the K=8 cutoff than non-transition chunks. Aggregate high-index energy is `0.0320` for transition chunks versus `0.00453` for non-transition chunks, about `7.1x` higher. The gripper channel is the clearest case: transition high-index energy is `0.0447`, while non-transition gripper high-index energy is `0.0` under the tau=1.0 labeling rule. This supports the contact-transition explanation for why a fixed low-pass representation can discard behaviorally important signal on Can.

## P1.c Can Sparse Retention Histogram

![Can sparse retention histogram](../../outputs/analysis/p1c_can_sparse_retention/figures/sparse_retention_histogram.png)

The retention histogram gives a more nuanced result than the simple hypothesis. Under the implemented sparse rule, which keeps the top-K temporal DCT bins by channel-summed coefficient energy, non-transition chunks select high-index bins more often by fraction: `26.2%` of retained bins have `k >= 8`, compared with `16.8%` for transition chunks. So sparse does not rescue Can simply because transition chunks choose high-frequency bins more frequently. A better reading is that sparse preserves whichever full-spectrum bins are large for each chunk; transition chunks have much larger high-frequency energy when it appears, especially in the gripper channel, while the selection histogram is also affected by low-frequency dominance and channel aggregation.

## P3 Can Smoothness vs Success

![Can smoothness vs success](../../outputs/analysis/p3_can_k_sweep/figures/can_smoothness_vs_success.png)

The sparse K sweep did not produce a clean monotonic smoothness-success tradeoff. Sparse K=4 was both smoothest and strongest in this single-seed sweep (`smoothness=0.0320`, success `0.96`), while K=8 and K=12 both landed at `0.92`, and K=16/raw was `0.94`. The important comparison is the low-frequency K=8 overlay: it has intermediate smoothness (`0.0500`) but much worse success (`0.62`), so the Can drop is not explained by smoothness alone. The failure mode is more specifically tied to fixed low-pass frequency choice.

## What This Means

- Sparse full-spectrum DCT is robust on Can across three seeds and recovers most of the raw-policy success that fixed low-pass loses.
- The explanatory evidence is conditional: transition chunks do carry more high-frequency energy, but sparse retention frequency is not higher for transition chunks under the bin-level top-K rule.
- Representation choice depends on task structure: Can benefits from adaptive full-spectrum preservation, Lift is saturated either way, and Push-T does not show the same advantage over low-pass smoothing.
