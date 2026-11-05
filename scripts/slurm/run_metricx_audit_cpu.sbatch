#!/bin/bash
#SBATCH --job-name=metricx_audit
#SBATCH --partition=m9
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/logs/metricx_audit_cpu_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENSSL_CONF=/dev/null
cd src
"${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}" -u audit_metricx_qe.py
