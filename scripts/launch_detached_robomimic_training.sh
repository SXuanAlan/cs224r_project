#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG_JSON [extra robomimic train.py args...]" >&2
  exit 2
fi

CONFIG_PATH="$1"
shift || true

STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="$(basename "${CONFIG_PATH}" .json)"
LOG_DIR="logs/detached"
mkdir -p "${LOG_DIR}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export WANDB_MODE="${WANDB_MODE:-online}"

LOG_PATH="${LOG_DIR}/${NAME}_${STAMP}.log"
PID_PATH="${LOG_DIR}/${NAME}_${STAMP}.pid"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "could not find conda.sh at ${CONDA_SH}; set CONDA_SH explicitly" >&2
  exit 1
fi

RUN_NAME="${NAME}_${STAMP}"

nohup setsid bash -lc "source '${CONDA_SH}' && conda activate cs224r && cd '${PWD}' && python -u robomimic/robomimic/scripts/train.py --config '${CONFIG_PATH}' --name '${RUN_NAME}' $*" \
  </dev/null > "${LOG_PATH}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_PATH}"
echo "started detached headless robomimic training"
echo "  pid: ${PID}"
echo "  log: ${LOG_PATH}"
echo "  pid_file: ${PID_PATH}"
echo "  run_name: ${RUN_NAME}"
echo "  env: MUJOCO_GL=${MUJOCO_GL} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM} WANDB_MODE=${WANDB_MODE} MPLBACKEND=${MPLBACKEND}"
