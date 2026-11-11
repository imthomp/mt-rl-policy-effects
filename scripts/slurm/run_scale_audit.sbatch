#!/bin/bash
#SBATCH --job-name=RL_scale_audit
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/logs/RL_scale_audit_%A_%a.out
# Policy-relative audits for the scale protocol (requires the frozen rule).
# 0 A en-de (7B), 1 A en-zh (7B), 2 B en-de (1.5B, new rewards), 3 B en-zh (1.5B, new rewards).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENSSL_CONF=/dev/null
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
RULE="$PWD/results/policy_relative_rule.json"; SHA=35f0a2276fa5daaea87f2987a176277593c5f5c17f65f23f504998798cac2821
NEW=$(python3 -c "import json;print(' '.join(json.load(open('results/grpo_scale_protocol.json'))['arm_B']['new_rewards']))")
cfg=(
  "A en_de German --policy-model Qwen/Qwen2.5-7B-Instruct"
  "A en_zh Chinese --policy-model Qwen/Qwen2.5-7B-Instruct"
  "B en_de German --rewards $NEW"
  "B en_zh Chinese --rewards $NEW"
)
read -r arm lp lang rest <<< "${cfg[$SLURM_ARRAY_TASK_ID]}"
cd src
"$PY" -u audit_policy_relative.py --target "$lang" --rule "$RULE" --rule-sha256 "$SHA" \
  --output "$PWD/../results/policy_relative_audit_scale_${arm}_${lp}.json" $rest
