#!/bin/bash
#SBATCH --job-name=judge_t4
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=artifacts/logs/llm_judge_t4_%A_%a.out

# T4 known-hack detection (results/llm_judge_protocol_amendment_1.json).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
JUDGE_PYTHON="${JUDGE_PYTHON:-${HOME}/.conda/envs/glm53/bin/python}"
export PATH="$(dirname "$JUDGE_PYTHON"):$PATH"
judges=(qwen2.5-32b gemma-2-27b gpt-oss-20b)
judge="${judges[$SLURM_ARRAY_TASK_ID]}"
cd src
"$JUDGE_PYTHON" -u run_llm_judge.py --judge "$judge" --tensor-parallel-size 2 \
  --items ../artifacts/llm_judge/t4_items.jsonl \
  --output "../artifacts/llm_judge/t4/${judge}.jsonl"
