#!/bin/bash
#SBATCH --job-name=judge_t4_qe
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=artifacts/logs/llm_judge_t4_qe_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"
models=(wmt22-cometkiwi-da wmt23-cometkiwi-da-xl XCOMET-XL)
model="${models[$SLURM_ARRAY_TASK_ID]}"
cd src
"$TRAINING_PYTHON" -u score_llm_judge_t4_qe.py --model-name "$model" \
  --output "../artifacts/llm_judge/t4_qe/${model}.jsonl"
