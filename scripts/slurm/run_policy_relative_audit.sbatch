#!/bin/bash
#SBATCH --job-name=RLMTpr_audit
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/logs/pr_audit_%A_%a.out
# Policy-relative audit. Array 0 German, 1 Spanish (retrospective calibration).
# Chinese: sbatch --array=2 --export=ALL,RULE_SHA256=<sha> (requires the frozen rule).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENSSL_CONF=/dev/null
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
langs=(German Spanish Chinese); codes=(en_de en_es en_zh)
lang=${langs[$SLURM_ARRAY_TASK_ID]}; code=${codes[$SLURM_ARRAY_TASK_ID]}
extra=()
if [[ "$lang" == Chinese ]]; then extra=(--rule "$PWD/results/policy_relative_rule.json" --rule-sha256 "$RULE_SHA256"); fi
cd src
"$PY" -u audit_policy_relative.py --target "$lang" --output "$PWD/../results/policy_relative_audit_${code}.json" "${extra[@]}"
