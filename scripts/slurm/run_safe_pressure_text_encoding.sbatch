#!/bin/bash
#SBATCH --job-name=RL_safe_encode
#SBATCH --partition=m9g
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:1
#SBATCH --time=00:45:00
#SBATCH --output=artifacts/logs/safe_pressure_encoding_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"
dataset_root="${MQM_DATA_ROOT:-${HOME}/projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation}"

"$TRAINING_PYTHON" -u src/encode_mqm_confirmation_candidates.py \
  --dataset-name wmt24_en_es_mqm \
  --source "$dataset_root/generalMT2024/mqm_generalMT2024_enes.tsv" \
  --manifest results/safe_pressure_manifest.json \
  --output artifacts/bandit/safe_pressure/wmt24_en_es_mqm_text_embeddings.npz
