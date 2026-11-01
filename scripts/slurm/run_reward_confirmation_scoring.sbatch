#!/bin/bash
#SBATCH --job-name=score_confirm
#SBATCH --partition=m13l
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-11%4
#SBATCH --output=artifacts/logs/reward_confirmation_score_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"
dataset_root="${MQM_DATA_ROOT:-${HOME}/projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation}"

seeds=(20261203 20261204 20261205)
conditions=(all_data diagnostic_filtered global_random_2 direction_matched_earlier_year)
seed_index=$((SLURM_ARRAY_TASK_ID / 4))
condition_index=$((SLURM_ARRAY_TASK_ID % 4))
seed="${seeds[$seed_index]}"
condition="${conditions[$condition_index]}"
model_root="artifacts/reward_models/confirmation/${seed}/models"
output_root="artifacts/reward_models/confirmation/${seed}/predictions"

"$TRAINING_PYTHON" -u src/score_mqm_confirmation_candidates.py \
  --dataset-name wmt24_ja_zh \
  --source "$dataset_root/generalMT2024/mqm_generalMT2024_jazh.tsv" \
  --manifest results/predictor_confirmation_manifest.json \
  --condition "$condition" \
  --model-root "$model_root" \
  --output "$output_root/wmt24_ja_zh/${condition}.jsonl"

"$TRAINING_PYTHON" -u src/score_mqm_confirmation_candidates.py \
  --dataset-name wmt23_zh_en \
  --source "$dataset_root/generalMT2023/zhen/mqm_generalMT2023_zhen.tsv" \
  --manifest results/predictor_confirmation_manifest.json \
  --condition "$condition" \
  --model-root "$model_root" \
  --output "$output_root/wmt23_zh_en/${condition}.jsonl"
