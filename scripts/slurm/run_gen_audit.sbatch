#!/bin/bash
#SBATCH --job-name=RL_gen_audit
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/logs/RL_gen_audit_%A_%a.out
# Policy-relative audits for the scale protocol (requires the frozen rule).
# 0 C en-de (1.5B, XXL rewards), 1 C en-zh (1.5B, XXL rewards), 2 D en-ja (7B), 3 D en-ru (7B).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENSSL_CONF=/dev/null
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
RULE="$PWD/results/policy_relative_rule.json"; SHA=35f0a2276fa5daaea87f2987a176277593c5f5c17f65f23f504998798cac2821
cfg=(
  "C en_de German --rewards xcomet_xxl_qe cometkiwi_23_xxl"
  "C en_zh Chinese --rewards xcomet_xxl_qe cometkiwi_23_xxl"
  "D en_ja Japanese --policy-model Qwen/Qwen2.5-7B-Instruct"
  "D en_ru Russian --policy-model Qwen/Qwen2.5-7B-Instruct"
)
read -r arm lp lang rest <<< "${cfg[$SLURM_ARRAY_TASK_ID]}"
cd src
"$PY" -u audit_policy_relative.py --target "$lang" --rule "$RULE" --rule-sha256 "$SHA" \
  --output "$PWD/../results/policy_relative_audit_gen_${arm}_${lp}.json" $rest
