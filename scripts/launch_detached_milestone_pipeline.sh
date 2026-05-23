#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
STAMP="${MILESTONE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs/detached"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/milestone_pipeline_${STAMP}.log"
PID_PATH="${LOG_DIR}/milestone_pipeline_${STAMP}.pid"
if [[ -f "${LOG_PATH}" ]]; then
  LOG_PATH="${LOG_DIR}/milestone_pipeline_${STAMP}_resume_$(date +%H%M%S).log"
  PID_PATH="${LOG_PATH%.log}.pid"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MILESTONE_STAMP="${STAMP}"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "could not find conda.sh at ${CONDA_SH}; set CONDA_SH explicitly" >&2
  exit 1
fi

nohup setsid bash -lc "source '${CONDA_SH}' && conda activate cs224r && cd '${PWD}' && scripts/run_milestone_pipeline.sh '${MODE}'" \
  </dev/null > "${LOG_PATH}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_PATH}"
echo "started detached milestone pipeline"
echo "  pid: ${PID}"
echo "  log: ${LOG_PATH}"
echo "  pid_file: ${PID_PATH}"
echo "  pipeline_dir: logs/pipeline/${STAMP}"
echo "  mode: ${MODE}"
