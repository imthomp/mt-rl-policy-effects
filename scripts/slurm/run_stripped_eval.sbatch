#!/bin/bash
#SBATCH --job-name=RLMTstripped_eval
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=artifacts/logs/stripped_eval_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENSSL_CONF=/dev/null
cd src
"${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}" -u eval_stripped_reference.py
