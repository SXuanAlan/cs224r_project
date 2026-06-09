# When Does DCT Help a Robot Policy?

**Action-Representation Trade-offs in Generative Imitation Learning**

CS224R: Deep Reinforcement Learning, Stanford University, Spring 2026

Alan Zhao and Juhyun Jung

## Method

For each action chunk `A` with horizon `H`, DCT variants transform each action
channel along time, mask coefficients, and reconstruct the chunk:

```text
Z = DCT(A)
A_hat = IDCT(mask * Z)
```

The main target variants differ only in how `mask` is chosen:

| Variant | Mask |
| --- | --- |
| Raw FM | No DCT transform |
| DCT Full-Freq | Keep every DCT coefficient |
| Sparse-DCT K | Keep the top-`K` coefficients by magnitude |
| Stochastic residual | Add a stochastic residual branch around the base target |
| Channel-aware Sparse-DCT | Apply sparse retention with per-channel structure |

All variants use the same Flow Matching policy family, observation interface,
EMA checkpointing, and receding-horizon evaluation protocol unless a task
requires a task-specific rollout script.

## Repository Layout

```text
cs224r_project/
|-- src/fgac/
|   |-- models/          # Flow Matching policy and policy IO helpers
|   |-- transforms/      # DCT and spectral transforms
|   |-- analysis/        # Frequency metrics and plotting helpers
|   `-- utils/           # Config and logging utilities
|-- scripts/
|   |-- train_flow_matching.py
|   |-- train_pusht_flow_matching.py
|   |-- train_maniskill_flow_matching.py
|   |-- eval_sim_rollout.py
|   |-- eval_pusht_rollout.py
|   |-- eval_maniskill_rollout.py
|   |-- run_frequency_analysis.py
|   |-- run_per_phase_spectrum.py
|   |-- run_sparse_retention_analysis.py
|   `-- summarize_checkpoint_eval.py
|-- configs/
|   |-- analysis/
|   `-- train/
`-- docs/reports/
    |-- final project report.pdf
    `-- figures/final_report/
```

## Installation

The robot environments use robomimic, robosuite v1.5.1, Push-T, and ManiSkill
depending on the task.

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git -b v1.5.1
pip install -e ./robosuite

git clone https://github.com/ARISE-Initiative/robomimic.git
pip install -e ./robomimic

pip install -r requirements.txt
bash scripts/download_milestone_datasets.sh
```

Robomimic datasets are expected under
`robomimic/datasets/<task>/ph/low_dim_v15.hdf5`. Push-T data is expected under
`data/pusht/pusht_cchi_v7_replay.zarr`.

## Reproducing Runs

Example robomimic training run:

```bash
python scripts/train_flow_matching.py --config configs/train/can_fm_raw_seed2.yaml
```

Example Push-T training run:

```bash
python scripts/train_pusht_flow_matching.py --config configs/train/pusht_fm_dct_sparse_k8_seed2.yaml
```

Example ManiSkill training run:

```bash
python scripts/train_maniskill_flow_matching.py --config configs/train/peg_insertion_side_fm_dct_sparse_k8_seed2.yaml
```

Evaluate robomimic checkpoints with:

```bash
bash scripts/run_flow_matching_eval.sh \
    outputs/train/<run>/checkpoints/<run>/<timestamp>/best.pt env 50
```

Outputs are written to `outputs/train/`, `outputs/eval/`, and
`outputs/analysis/`. These generated outputs are not required for reading the
submitted final report because the final PDF and figures are stored under
`docs/reports/`.

## Final Figures

The final report uses the figure assets in `docs/reports/figures/final_report/`,
including:

- `method_comparison.png`
- `fig2_k_sweep.png`
- `fig3_energy_decoupling.png`
- `fig4_residual_outcome_stochastic.png`
- `fig5_demo_spectrum_per_task.png`
- `cross_task_spectra.png`
- `can_spectral_heatmaps.png`
- `square_spectral_heatmaps.png`
- `peg_insertion_side_spectral_heatmaps.png`
- `can_phase_spectrum_aggregate.png`
- `can_phase_spectrum_per_channel.png`
- `square_phase_spectrum_aggregate.png`
- `square_phase_spectrum_per_channel.png`

## Limitations

- Results are simulator-only.
- Reported policy evaluations use one seed and 50 rollout episodes.
- Success rates are empirical and should be read as task-level evidence, not as
  formal statistical guarantees.
