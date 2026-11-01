#!/bin/bash
#SBATCH --job-name=wmt25_reward_score
#SBATCH --partition=m9g
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:1
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=artifacts/logs/wmt25_reward_score_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"
WMT25_ARCHIVE="${WMT25_ARCHIVE:-data/reward_da/wmt25-mqm.zip}"

conditions=(all_data diagnostic_filtered global_random_2 direction_matched_earlier_year)
condition="${conditions[$SLURM_ARRAY_TASK_ID]}"

"$TRAINING_PYTHON" -u src/score_wmt25_reward_candidates.py \
  --condition "$condition" \
  --archive "$WMT25_ARCHIVE" \
  --model-root artifacts/reward_models/development \
  --output "artifacts/reward_models/wmt25_predictions/${condition}.jsonl"
