#!/bin/bash
#SBATCH --job-name=judge_qe
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=artifacts/logs/judge_validation_qe_%A_%a.out

# External QE rewards needed by the LLM-judge validation (T1 reference points,
# T3 reward cells).  Skips any output that already exists.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"
root="${MQM_DATA_ROOT:-${HOME}/projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation}"
out="artifacts/reward_models/confirmation/external/predictions"

# task: model|dataset|tsv|manifest
tasks=(
  "wmt22-cometkiwi-da|wmt24_en_es_mqm|generalMT2024/mqm_generalMT2024_enes.tsv|results/safe_pressure_manifest.json"
  "wmt23-cometkiwi-da-xl|wmt24_en_es_mqm|generalMT2024/mqm_generalMT2024_enes.tsv|results/safe_pressure_manifest.json"
  "XCOMET-XL|wmt24_en_es_mqm|generalMT2024/mqm_generalMT2024_enes.tsv|results/safe_pressure_manifest.json"
  "XCOMET-XL|wmt23_zh_en|generalMT2023/zhen/mqm_generalMT2023_zhen.tsv|results/predictor_confirmation_manifest.json"
  "XCOMET-XL|wmt24_ja_zh|generalMT2024/mqm_generalMT2024_jazh.tsv|results/predictor_confirmation_manifest.json"
)
IFS='|' read -r model dataset tsv manifest <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"
target="$out/$dataset/${model}.jsonl"
if [[ -f "$target" ]]; then echo "exists: $target"; exit 0; fi

"$TRAINING_PYTHON" -u src/score_mqm_confirmation_comet.py \
  --model-name "$model" \
  --dataset-name "$dataset" \
  --source "$root/$tsv" \
  --manifest "$manifest" \
  --output "$target" \
  --batch-size 16
