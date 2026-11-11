#!/bin/bash
#SBATCH --job-name=RL_mt_text_pred
#SBATCH --partition=m9g
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:1
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/logs/text_policy_predictors_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"

"$TRAINING_PYTHON" -u src/analyze_text_policy_predictors.py
