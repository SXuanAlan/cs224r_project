#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
STAMP="${MILESTONE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PIPELINE_DIR="logs/pipeline/${STAMP}"
MARKER_DIR="${PIPELINE_DIR}/markers"
mkdir -p "${MARKER_DIR}"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "could not find conda.sh at ${CONDA_SH}; set CONDA_SH explicitly" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate cs224r

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_step() {
  local name="$1"
  shift
  local marker="${MARKER_DIR}/${name}.done"
  local log_path="${PIPELINE_DIR}/${name}.log"
  if [[ -f "${marker}" ]]; then
    log "skip ${name}"
    return 0
  fi
  log "start ${name}"
  {
    echo "COMMAND: $*"
    "$@"
  } 2>&1 | tee "${log_path}"
  touch "${marker}"
  log "done ${name}"
}

latest_ckpt() {
  local base="$1"
  local latest
  latest="$(find "${base}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "${latest}" || ! -f "${latest}/best.pt" ]]; then
    echo "missing checkpoint under ${base}" >&2
    return 1
  fi
  echo "${latest}/best.pt"
}

ensure_data() {
  if [[ ! -f robomimic/datasets/can/ph/low_dim_v15.hdf5 ]]; then
    run_step download_can scripts/download_milestone_datasets.sh can
  fi
  if [[ ! -d data/pusht/pusht_cchi_v7_replay.zarr ]]; then
    run_step setup_pusht_deps scripts/setup_pusht_deps.sh
    run_step download_pusht python scripts/download_pusht_dataset.py
  fi
}

run_frequency() {
  run_step freq_lift python scripts/run_frequency_analysis.py --config configs/analysis/frequency_diagnostic.yaml
  run_step freq_can python scripts/run_frequency_analysis.py --config configs/analysis/can_frequency_diagnostic.yaml
  run_step freq_pusht python scripts/run_pusht_frequency_analysis.py --config configs/analysis/pusht_frequency_diagnostic.yaml
}

train_eval_robomimic() {
  local name="$1"
  local config="$2"
  local ckpt_base="$3"
  run_step "train_${name}" python -u scripts/train_flow_matching.py --config "${config}"
  local ckpt
  ckpt="$(latest_ckpt "${ckpt_base}")"
  run_step "eval_${name}_env50" scripts/run_flow_matching_eval.sh "${ckpt}" env 50
  run_step "eval_${name}_dataset10" scripts/run_flow_matching_eval.sh "${ckpt}" dataset 10
}

train_eval_pusht() {
  local name="$1"
  local config="$2"
  local ckpt_base="$3"
  run_step "train_${name}" python -u scripts/train_pusht_flow_matching.py --config "${config}"
  local ckpt
  ckpt="$(latest_ckpt "${ckpt_base}")"
  run_step "eval_${name}_env50" scripts/run_pusht_eval.sh "${ckpt}" 50
}

run_reports() {
  run_step generate_reports python scripts/generate_milestone_report.py
}

log "milestone pipeline mode=${MODE} stamp=${STAMP}"
ensure_data
run_frequency

if [[ "${MODE}" == "analysis-only" ]]; then
  run_reports
  exit 0
fi

train_eval_robomimic lift_raw configs/train/fm_raw.yaml outputs/train/raw_fm/checkpoints
train_eval_robomimic lift_dct configs/train/fm_dct_lowfreq.yaml outputs/train/dct_lowfreq_k8/checkpoints
train_eval_robomimic can_raw configs/train/can_fm_raw.yaml outputs/train/can_raw_fm/checkpoints
train_eval_robomimic can_dct configs/train/can_fm_dct_lowfreq.yaml outputs/train/can_dct_lowfreq_k8/checkpoints
train_eval_pusht pusht_raw configs/train/pusht_fm_raw.yaml outputs/train/pusht_raw_fm/checkpoints
train_eval_pusht pusht_dct configs/train/pusht_fm_dct_lowfreq.yaml outputs/train/pusht_dct_lowfreq_k8/checkpoints
run_reports

log "milestone pipeline complete"
