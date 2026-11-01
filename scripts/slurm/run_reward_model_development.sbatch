#!/bin/bash
#SBATCH --job-name=reward_model_dev
#SBATCH --partition=m13l
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=artifacts/logs/reward_model_dev_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"
arrow_dir="${DA_ARROW_DIR:-${HOME}/.cache/huggingface/datasets/RicardoRei___wmt-da-human-evaluation/default/0.0.0/301de385bf05b0c00a8f4be74965e186164dd425}"

conditions=(all_data diagnostic_filtered global_random_2 direction_matched_earlier_year)
condition="${conditions[$SLURM_ARRAY_TASK_ID]}"

"$TRAINING_PYTHON" -u src/train_da_reward_model.py \
  --condition "$condition" \
  --arrow-dir "$arrow_dir" \
  --indices data/reward_da/development_indices.json \
  --output-dir "artifacts/reward_models/development/${condition}"
