# Output Layout

Experiment outputs are grouped by experiment type and method, instead of mixing all metrics, checkpoints, figures, and videos in one flat directory.

```text
outputs/
  analysis/
    frequency_diagnostic/
      metrics/
      figures/

  train/
    raw_fm/
      metrics/
      checkpoints/
      videos/

    dct_lowfreq_k8/
      metrics/
      checkpoints/
      videos/

  eval/
    simulation/
      <checkpoint_run_name>/
        simulation_rollout.mp4
        simulation_rollout.json
```

Logs follow the same grouping where possible:

```text
logs/
  analysis/
    frequency_diagnostic/

  train/
    raw_fm/
    dct_lowfreq_k8/

  detached/
```

`logs/detached/` stores stdout/stderr and PID files for detached headless launches.

