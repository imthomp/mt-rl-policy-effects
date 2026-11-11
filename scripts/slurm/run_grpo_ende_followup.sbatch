#!/bin/bash
#SBATCH --job-name=RLMTgrpo_followup
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#SBATCH --output=artifacts/logs/grpo_followup_%A_%a.out
# Exploratory en-de follow-up (results/grpo_ende_protocol_amendment_2.json):
#   0-2 CometKiwi-23-XL seeds 0-2, 1000 steps, unguarded; 3-5 same with off-target-language guard.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENSSL_CONF=/dev/null
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
runs=("0 none" "1 none" "2 none" "0 langid" "1 langid" "2 langid")
read -r seed guard <<< "${runs[$SLURM_ARRAY_TASK_ID]}"
guard_flag=(); name="cometkiwi_23_xl_steps1000_seed${seed}"
if [[ "$guard" == langid ]]; then guard_flag=(--langid-guard); name="cometkiwi_23_xl_langid_steps1000_seed${seed}"; fi
# Checkpoints (~6 GB each) go to /nobackup; home quota is 2 TB. Small eval files are copied back.
out="/nobackup/archive/usr/it238/reinforcement-learning/grpo_ende_followup/${name}"
keep="$PWD/artifacts/grpo_ende_followup/${name}"
cd src
"$PY" -u train_grpo_pilot.py --protocol-name ende --reward cometkiwi_23_xl --seed "$seed" --steps 1000 "${guard_flag[@]}" --output-dir "$out"
"$PY" -u eval_grpo_pilot.py --protocol-name ende --steps 1000 --run-dir "$out"
mkdir -p "$keep" && cp "$out"/eval_summary.json "$out"/eval_outputs.jsonl "$out"/pilot_run.json "$keep"/
