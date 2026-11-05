#!/bin/bash
#SBATCH --job-name=grpo_rep
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
name="${reward}$([[ $guarded == 1 ]] && echo _guarded)_seed${seed}"
out="$PWD/artifacts/grpo_replication/${name}"
cd src
"$PY" -u train_grpo_pilot.py --reward "$reward" --seed "$seed" $([[ $guarded == 1 ]] && echo --guarded) --output-dir "$out"
"$PY" -u eval_grpo_pilot.py --run-dir "$out"
