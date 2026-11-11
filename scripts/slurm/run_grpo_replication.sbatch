#!/bin/bash
#SBATCH --job-name=MTRLgrpo_rep
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-8
#SBATCH --output=artifacts/logs/grpo_rep_%A_%a.out

# results/grpo_replication_protocol.json: tasks 0-5 replication, 6-8 guarded XCOMET.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
export OPENSSL_CONF=/dev/null
runs=(
  "xcomet_xl_qe 1 0" "xcomet_xl_qe 2 0"
  "cometkiwi_22 1 0" "cometkiwi_22 2 0"
  "cometkiwi_23_xl 1 0" "cometkiwi_23_xl 2 0"
  "xcomet_xl_qe 0 1" "xcomet_xl_qe 1 1" "xcomet_xl_qe 2 1"
)
read -r reward seed guarded <<< "${runs[$SLURM_ARRAY_TASK_ID]}"
guard_flag=()
suffix=""
if [[ "$guarded" == 1 ]]; then guard_flag=(--guarded); suffix="_guarded"; fi
name="${reward}${suffix}_seed${seed}"
# Checkpoints (~6 GB each) go to /nobackup; home quota is 2 TB. Small eval files are copied back.
out="/nobackup/archive/usr/it238/reinforcement-learning/grpo_replication/${name}"
keep="$PWD/artifacts/grpo_replication/${name}"
cd src
"$PY" -u train_grpo_pilot.py --reward "$reward" --seed "$seed" "${guard_flag[@]}" --output-dir "$out"
"$PY" -u eval_grpo_pilot.py --run-dir "$out"
mkdir -p "$keep" && cp "$out"/eval_summary.json "$out"/eval_outputs.jsonl "$out"/pilot_run.json "$keep"/
