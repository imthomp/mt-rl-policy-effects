#!/bin/bash
#SBATCH --job-name=RL_margin_audit
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/logs/RL_margin_audit_%A_%a.out
# Policy-relative audits for the scale protocol (requires the frozen rule).
# Margin-test candidate audits (results/margin_test_protocol.json): 0 7B de XXL, 1 7B es XL+XXL, 2 7B zh XXL.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENSSL_CONF=/dev/null
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
RULE="$PWD/results/policy_relative_rule.json"; SHA=35f0a2276fa5daaea87f2987a176277593c5f5c17f65f23f504998798cac2821
cfg=(
  "M en_de German --policy-model Qwen/Qwen2.5-7B-Instruct --rewards xcomet_xxl_qe"
  "M en_es Spanish --policy-model Qwen/Qwen2.5-7B-Instruct --rewards xcomet_xl_qe xcomet_xxl_qe"
  "M en_zh Chinese --policy-model Qwen/Qwen2.5-7B-Instruct --rewards xcomet_xxl_qe"
)
read -r arm lp lang rest <<< "${cfg[$SLURM_ARRAY_TASK_ID]}"
cd src
"$PY" -u audit_policy_relative.py --target "$lang" --rule "$RULE" --rule-sha256 "$SHA" \
  --output "$PWD/../results/policy_relative_audit_margin_${arm}_${lp}.json" $rest
