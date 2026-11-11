#!/bin/bash
#SBATCH --job-name=RL_safe_calibrate
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/logs/safe_pressure_calibration_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"

"$TRAINING_PYTHON" -u src/calibrate_safe_pressure_selector.py
