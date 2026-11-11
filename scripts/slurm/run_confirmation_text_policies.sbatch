#!/bin/bash
#SBATCH --job-name=RL_policy_confirm
#SBATCH --partition=m9g
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:1
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=artifacts/logs/confirmation_policy_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"
datasets=(wmt24_ja_zh wmt23_zh_en)
dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"

"$TRAINING_PYTHON" -u src/run_confirmation_text_policies.py \
  --dataset "$dataset" \
  --output "results/confirmation_policy_${dataset}.json" \
  --probabilities-output "artifacts/bandit/confirmation/${dataset}_policy_probabilities.npz"
