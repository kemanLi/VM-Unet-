#!/usr/bin/env bash
set -Eeuo pipefail

project_root="${1:-/root/autodl-tmp/VM-UNet}"
venv_root="${VMUNET_VENV:-/root/autodl-tmp/venvs/vmunet}"

cd "$project_root"
source "$venv_root/bin/activate"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "[$(date '+%F %T')] Sobel operator ablation preflight"
python -m unittest \
    experiments.sobel_operator_ablation.tests.test_sobel_guidance -v
python -u experiments/sobel_operator_ablation/compute_q_stats.py
python -u experiments/sobel_operator_ablation/smoke_model.py \
    --dataset DRIVE \
    --operator s3_d2

echo "[$(date '+%F %T')] 01/10 DRIVE s3_d2"
python -u experiments/sobel_operator_ablation/run_one.py --dataset DRIVE --operator s3_d2

echo "[$(date '+%F %T')] 02/10 DRIVE s3_d4"
python -u experiments/sobel_operator_ablation/run_one.py --dataset DRIVE --operator s3_d4

echo "[$(date '+%F %T')] 03/10 DRIVE s5_d2"
python -u experiments/sobel_operator_ablation/run_one.py --dataset DRIVE --operator s5_d2

echo "[$(date '+%F %T')] 04/10 DRIVE s5_d4"
python -u experiments/sobel_operator_ablation/run_one.py --dataset DRIVE --operator s5_d4

echo "[$(date '+%F %T')] 05/10 DRIVE s5_d8"
python -u experiments/sobel_operator_ablation/run_one.py --dataset DRIVE --operator s5_d8

echo "[$(date '+%F %T')] 06/10 STARE s3_d2"
python -u experiments/sobel_operator_ablation/run_one.py --dataset STARE --operator s3_d2

echo "[$(date '+%F %T')] 07/10 STARE s3_d4"
python -u experiments/sobel_operator_ablation/run_one.py --dataset STARE --operator s3_d4

echo "[$(date '+%F %T')] 08/10 STARE s5_d2"
python -u experiments/sobel_operator_ablation/run_one.py --dataset STARE --operator s5_d2

echo "[$(date '+%F %T')] 09/10 STARE s5_d4"
python -u experiments/sobel_operator_ablation/run_one.py --dataset STARE --operator s5_d4

echo "[$(date '+%F %T')] 10/10 STARE s5_d8"
python -u experiments/sobel_operator_ablation/run_one.py --dataset STARE --operator s5_d8

echo "[$(date '+%F %T')] All Sobel operator ablations completed"
