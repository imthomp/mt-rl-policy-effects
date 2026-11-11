#!/bin/bash
#SBATCH --job-name=RL_grpo_enzh
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=artifacts/logs/grpo_enzh_%A_%a.out
# en-zh prospective test of results/policy_relative_rule.json
# (predictions: results/grpo_enzh_predictions.json). 4 rewards x seeds 0-2, 500 steps.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENSSL_CONF=/dev/null
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
rewards=(xcomet_xl_qe cometkiwi_22 cometkiwi_23_xl metricx_24_qe)
reward=${rewards[$((SLURM_ARRAY_TASK_ID % 4))]}
seed=$((SLURM_ARRAY_TASK_ID / 4))
# Checkpoints (~6 GB each) go to /nobackup; home quota is 2 TB. Small eval files are copied back.
out="/nobackup/archive/usr/it238/reinforcement-learning/grpo_rule_tests/en_zh/${reward}_seed${seed}"
keep="$PWD/artifacts/grpo_rule_tests/en_zh/${reward}_seed${seed}"
cd src
"$PY" -u train_grpo_pilot.py --protocol-name enzh --reward "$reward" --seed "$seed" --output-dir "$out"
"$PY" -u eval_grpo_pilot.py --protocol-name enzh --run-dir "$out"
mkdir -p "$keep" && cp "$out"/eval_summary.json "$out"/eval_outputs.jsonl "$out"/pilot_run.json "$keep"/
