#!/usr/bin/env bash
set -euo pipefail

TASKS="${*:-lift can}"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "could not find conda.sh at ${CONDA_SH}; set CONDA_SH explicitly" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate cs224r

python robomimic/robomimic/scripts/download_datasets.py \
  --tasks ${TASKS} \
  --dataset_types ph \
  --hdf5_types low_dim
