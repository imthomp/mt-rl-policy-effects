#!/bin/bash
#SBATCH --job-name=comet_confirm
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=artifacts/logs/reward_confirmation_comet_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"
dataset_root="${MQM_DATA_ROOT:-${HOME}/projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation}"

models=(wmt22-cometkiwi-da wmt23-cometkiwi-da-xl)
model="${models[$SLURM_ARRAY_TASK_ID]}"
output_root="artifacts/reward_models/confirmation/external/predictions"

"$TRAINING_PYTHON" -u src/score_mqm_confirmation_comet.py \
  --model-name "$model" \
  --dataset-name wmt24_ja_zh \
  --source "$dataset_root/generalMT2024/mqm_generalMT2024_jazh.tsv" \
  --manifest results/predictor_confirmation_manifest.json \
  --output "$output_root/wmt24_ja_zh/${model}.jsonl"

"$TRAINING_PYTHON" -u src/score_mqm_confirmation_comet.py \
  --model-name "$model" \
  --dataset-name wmt23_zh_en \
  --source "$dataset_root/generalMT2023/zhen/mqm_generalMT2023_zhen.tsv" \
  --manifest results/predictor_confirmation_manifest.json \
  --output "$output_root/wmt23_zh_en/${model}.jsonl"
