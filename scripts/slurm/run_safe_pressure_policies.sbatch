#!/bin/bash
#SBATCH --job-name=safe_policy
#SBATCH --partition=m9g
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:1
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/logs/safe_pressure_policy_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"

"$TRAINING_PYTHON" -u src/run_safe_pressure_policies.py
