#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG_PATH [extra train_flow_matching.py args...]" >&2
  exit 2
fi

CONFIG_PATH="$1"
shift || true

STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="$(basename "${CONFIG_PATH}" .yaml)"
LOG_DIR="logs/detached"
mkdir -p "${LOG_DIR}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

LOG_PATH="${LOG_DIR}/${NAME}_${STAMP}.log"
PID_PATH="${LOG_DIR}/${NAME}_${STAMP}.pid"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "could not find conda.sh at ${CONDA_SH}; set CONDA_SH explicitly" >&2
  exit 1
fi

nohup setsid bash -lc "source '${CONDA_SH}' && conda activate cs224r && cd '${PWD}' && python -u scripts/train_flow_matching.py --config '${CONFIG_PATH}' $*" \
  </dev/null > "${LOG_PATH}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_PATH}"
echo "started detached headless training"
echo "  pid: ${PID}"
echo "  log: ${LOG_PATH}"
echo "  pid_file: ${PID_PATH}"
echo "  env: MUJOCO_GL=${MUJOCO_GL} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM} WANDB_MODE=${WANDB_MODE:-from_config} MPLBACKEND=${MPLBACKEND}"
