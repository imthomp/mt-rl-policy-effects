#!/bin/bash
#SBATCH --job-name=RL_smoke_scale
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=artifacts/logs/RL_smoke_scale_%A_%a.out
# Resource smoke tests only (2 GRPO steps, output to /tmp, never a result).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENSSL_CONF=/dev/null
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
cfgs=(
  "xcomet_xl_qe --policy-model Qwen/Qwen2.5-7B-Instruct --lora-r 32 --learning-rate 2e-5"
  "cometkiwi_23_xl --policy-model Qwen/Qwen2.5-7B-Instruct --lora-r 32 --learning-rate 2e-5"
  "metricx_24_xxl_qe"
  "xcomet_xxl_qe"
)
read -r reward rest <<< "${cfgs[$SLURM_ARRAY_TASK_ID]}"
cd src
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/usr/bin/time -v "$PY" -u train_grpo_pilot.py --protocol-name ende --reward "$reward" $rest --smoke --output-dir /tmp/unused 2>&1 | grep -v -E 'Warning|warn' | tail -40
nvidia-smi --query-gpu=memory.used --format=csv,noheader
