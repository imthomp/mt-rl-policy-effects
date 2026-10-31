#!/bin/bash
#SBATCH --job-name=da_diagnostic_metrics
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=artifacts/logs/da_diagnostic_metrics_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
SCORING_PYTHON="${SCORING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"

case "$SLURM_ARRAY_TASK_ID" in
  0) metric=comet; batch_size=32 ;;
  1) metric=bleurt; batch_size=64 ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

"$SCORING_PYTHON" -u src/score_da_diagnostic_sample.py \
  --metric "$metric" \
  --input data/reward_da/diagnostic_sample.jsonl \
  --output "artifacts/diagnostics/da_diagnostic_${metric}.jsonl" \
  --batch-size "$batch_size"
