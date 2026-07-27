#!/usr/bin/env bash
set -Eeuo pipefail

project_root="${1:-/root/autodl-tmp/VM-UNet}"
venv_root="${VMUNET_VENV:-/root/autodl-tmp/venvs/vmunet}"

cd "$project_root"
source "$venv_root/bin/activate"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p baseline_logs

run_baseline() {
    local dataset_name="$1"
    local start_time
    local log_path

    start_time="$(date +%Y%m%d_%H%M%S)"
    log_path="baseline_logs/${dataset_name}_baseline_${start_time}.log"

    echo "[$(date '+%F %T')] Starting ${dataset_name} baseline"
    echo "Training log: ${project_root}/${log_path}"

    python -u train.py \
        --dataset "$dataset_name" \
        --model vmunet \
        --initialization scratch \
        --run-tag baseline \
        >"$log_path" 2>&1

    echo "[$(date '+%F %T')] Completed ${dataset_name} baseline"
}

run_baseline DRIVE
run_baseline STARE

echo "[$(date '+%F %T')] All retinal baselines completed"
