#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT [n_rollouts]" >&2
  exit 2
fi

CHECKPOINT="$1"
N_ROLLOUTS="${2:-50}"
MAX_VIDEOS="${MAX_VIDEOS:-5}"
RUN_ID="$(basename "$(dirname "${CHECKPOINT}")")"
OUTPUT_DIR="outputs/eval/${RUN_ID}/env"
OUTPUT="${OUTPUT_DIR}/simulation_rollout.mp4"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "could not find conda.sh at ${CONDA_SH}; set CONDA_SH explicitly" >&2
  exit 1
fi

export MPLBACKEND="${MPLBACKEND:-Agg}"

source "${CONDA_SH}"
conda activate cs224r

python -u scripts/eval_pusht_rollout.py \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  --n-rollouts "${N_ROLLOUTS}" \
  --max-videos "${MAX_VIDEOS}"

python -u scripts/summarize_checkpoint_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --rollout-json "${OUTPUT_DIR}/simulation_rollout.json" \
  --output "${OUTPUT_DIR}/metrics.json"
