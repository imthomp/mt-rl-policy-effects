#!/bin/bash
#SBATCH --job-name=judge_v2
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=artifacts/logs/llm_judge_v2_%A_%a.out

# Judge validation v2 (results/llm_judge_protocol_v2.json): tasks 0-2 validation, 3-5 T4.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
JUDGE_PYTHON="${JUDGE_PYTHON:-${HOME}/.conda/envs/glm53/bin/python}"
export PATH="$(dirname "$JUDGE_PYTHON"):$PATH"
PROTOCOL_SHA=c56dd6a38e4545b471d4ab9cc63504999137de6bdb004b4d1d29bf0655fd75c5
ITEMS_SHA=409bf41b47e306cea0ce75cc32b6110bf94cc3d6dc52985001ad99f3f356e12a
judges=(qwen2.5-32b gemma-2-27b gpt-oss-20b)
judge="${judges[$((SLURM_ARRAY_TASK_ID % 3))]}"
cd src
if (( SLURM_ARRAY_TASK_ID < 3 )); then
  "$JUDGE_PYTHON" -u run_llm_judge.py --judge "$judge" --tensor-parallel-size 2 \
    --protocol ../results/llm_judge_protocol_v2.json --protocol-sha256 "$PROTOCOL_SHA" \
    --output "../artifacts/llm_judge/v2/validation/${judge}.jsonl"
else
  "$JUDGE_PYTHON" -u run_llm_judge.py --judge "$judge" --tensor-parallel-size 2 \
    --protocol ../results/llm_judge_protocol_v2.json --protocol-sha256 "$PROTOCOL_SHA" \
    --items ../artifacts/llm_judge/v2/t4_items.jsonl --items-sha256 "$ITEMS_SHA" \
    --output "../artifacts/llm_judge/v2/t4/${judge}.jsonl"
fi
