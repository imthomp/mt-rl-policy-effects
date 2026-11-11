#!/bin/bash
#SBATCH --job-name=RL_safe_score
#SBATCH --partition=m13l
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --array=0-14%4
#SBATCH --output=artifacts/logs/safe_pressure_score_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"
dataset_root="${MQM_DATA_ROOT:-${HOME}/projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation}"

readarray -t task < <("$TRAINING_PYTHON" - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
row=json.load(open('results/safe_pressure_reward_plan.json'))['tasks'][int(sys.argv[1])]
print(row['seed'])
print(row['training_condition'])
PY
)
seed="${task[0]}"
condition="${task[1]}"

"$TRAINING_PYTHON" -u src/score_mqm_confirmation_candidates.py \
  --dataset-name wmt24_en_es_mqm \
  --source "$dataset_root/generalMT2024/mqm_generalMT2024_enes.tsv" \
  --manifest results/safe_pressure_manifest.json \
  --condition "$condition" \
  --model-root "artifacts/reward_models/safe_pressure/${seed}/models" \
  --output "artifacts/reward_models/safe_pressure/${seed}/predictions/wmt24_en_es_mqm/${condition}.jsonl"
