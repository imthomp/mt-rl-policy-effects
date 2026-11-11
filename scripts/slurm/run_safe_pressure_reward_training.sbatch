#!/bin/bash
#SBATCH --job-name=RL_safe_reward
#SBATCH --partition=m13l
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --array=0-14%4
#SBATCH --output=artifacts/logs/safe_pressure_reward_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"
arrow_dir="${DA_ARROW_DIR:-${HOME}/.cache/huggingface/datasets/RicardoRei___wmt-da-human-evaluation/default/0.0.0/301de385bf05b0c00a8f4be74965e186164dd425}"

readarray -t task < <("$TRAINING_PYTHON" - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
plan=json.load(open('results/safe_pressure_reward_plan.json'))
row=plan['tasks'][int(sys.argv[1])]
print(row['seed'])
print(row['training_condition'])
print(row['output_dir'])
PY
)
seed="${task[0]}"
condition="${task[1]}"
output_dir="${task[2]}"

"$TRAINING_PYTHON" -u src/train_da_reward_model.py \
  --condition "$condition" \
  --arrow-dir "$arrow_dir" \
  --indices data/reward_da/development_indices.json \
  --output-dir "$output_dir" \
  --seed "$seed"
