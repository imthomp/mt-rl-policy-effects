#!/bin/bash
#SBATCH --job-name=encode_confirm
#SBATCH --partition=m9g
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=artifacts/logs/confirmation_encoding_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"
dataset_root="${MQM_DATA_ROOT:-${HOME}/projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation}"

names=(wmt24_ja_zh wmt23_zh_en)
sources=(generalMT2024/mqm_generalMT2024_jazh.tsv generalMT2023/zhen/mqm_generalMT2023_zhen.tsv)
name="${names[$SLURM_ARRAY_TASK_ID]}"
source="${sources[$SLURM_ARRAY_TASK_ID]}"

"$TRAINING_PYTHON" -u src/encode_mqm_confirmation_candidates.py \
  --dataset-name "$name" \
  --source "$dataset_root/$source" \
  --manifest results/predictor_confirmation_manifest.json \
  --output "artifacts/bandit/confirmation/${name}_text_embeddings.npz"
