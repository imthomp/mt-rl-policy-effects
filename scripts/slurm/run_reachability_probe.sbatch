#!/bin/bash
#SBATCH --job-name=RL_reach_probe
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --array=0-6
#SBATCH --output=artifacts/logs/RL_reach_probe_%A_%a.out
# Reachability probe (results/reachability_probe_protocol.json): initial-policy rollouts, TRL sampling.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENSSL_CONF=/dev/null
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
cfg=("Qwen/Qwen2.5-1.5B-Instruct German" "Qwen/Qwen2.5-1.5B-Instruct Spanish" "Qwen/Qwen2.5-1.5B-Instruct Chinese"
     "Qwen/Qwen2.5-7B-Instruct German" "Qwen/Qwen2.5-7B-Instruct Chinese" "Qwen/Qwen2.5-7B-Instruct Japanese" "Qwen/Qwen2.5-7B-Instruct Russian")
read -r policy target <<< "${cfg[$SLURM_ARRAY_TASK_ID]}"
cd src
"$PY" -u probe_reachability.py --policy "$policy" --target "$target"
