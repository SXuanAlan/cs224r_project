#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT [reset_mode] [n_rollouts]" >&2
  echo "  reset_mode: env or dataset (default: env)" >&2
  echo "  n_rollouts: default 50 for env, 10 for dataset" >&2
  exit 2
fi

CHECKPOINT="$1"
RESET_MODE="${2:-env}"
if [[ "${RESET_MODE}" == "env" ]]; then
  N_ROLLOUTS="${3:-50}"
  MAX_VIDEOS="${MAX_VIDEOS:-5}"
else
  N_ROLLOUTS="${3:-10}"
  MAX_VIDEOS="${MAX_VIDEOS:-10}"
fi

RUN_ID="$(basename "$(dirname "${CHECKPOINT}")")"
OUTPUT_DIR="outputs/eval/${RUN_ID}/${RESET_MODE}"
OUTPUT="${OUTPUT_DIR}/simulation_rollout.mp4"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "could not find conda.sh at ${CONDA_SH}; set CONDA_SH explicitly" >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

source "${CONDA_SH}"
conda activate cs224r

python -u scripts/eval_sim_rollout.py \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  --reset-mode "${RESET_MODE}" \
  --n-rollouts "${N_ROLLOUTS}" \
  --max-videos "${MAX_VIDEOS}"

python -u scripts/summarize_checkpoint_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --rollout-json "${OUTPUT_DIR}/simulation_rollout.json" \
  --output "${OUTPUT_DIR}/metrics.json"
