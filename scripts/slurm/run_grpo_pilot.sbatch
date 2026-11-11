#!/bin/bash
#SBATCH --job-name=RLMTgrpo_pilot
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=artifacts/logs/grpo_pilot_%A_%a.out

# GRPO pilot (results/grpo_pilot_protocol.json): train then evaluate one reward per task.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
# uv-managed CPython bundles OpenSSL that cannot parse this cluster's system config; jobs are offline.
export OPENSSL_CONF=/dev/null
rewards=(xcomet_xl_qe cometkiwi_22 cometkiwi_23_xl)
reward="${rewards[$SLURM_ARRAY_TASK_ID]}"
out="$PWD/artifacts/grpo_pilot/${SMOKE:+smoke_}${reward}"
cd src
"$PY" -u train_grpo_pilot.py --reward "$reward" --output-dir "$out" ${SMOKE:+--smoke}
if [[ -z "${SMOKE:-}" ]]; then "$PY" -u eval_grpo_pilot.py --run-dir "$out"; fi
