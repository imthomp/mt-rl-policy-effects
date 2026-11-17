#!/bin/bash
#SBATCH --job-name=RL_smoke_xxl2gpu
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=artifacts/logs/RL_smoke_xxl2gpu_%A_%a.out
# Resource smoke only: 1.5B policy on GPU 0, XXL COMET reward on GPU 1 (REWARD_DEVICE=1).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENSSL_CONF=/dev/null
export REWARD_DEVICE=1
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
rewards=(xcomet_xxl_qe cometkiwi_23_xxl)
cd src
"$PY" -u train_grpo_pilot.py --protocol-name ende --reward "${rewards[$SLURM_ARRAY_TASK_ID]}" --smoke --output-dir /tmp/unused 2>&1 | grep -v -E 'Warning|warn' | tail -25
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
