#!/bin/bash
#SBATCH --job-name=tatoeba_policy_cal
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/tatoeba_policy_cal_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLLB_ALLOCATOR_LIMIT_GIB="${NLLB_ALLOCATOR_LIMIT_GIB:-72}"
export NLLB_MODEL_DIR="${NLLB_MODEL_DIR:-${HOME}/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d}"
NLLB_PYTHON="${NLLB_PYTHON:-${HOME}/.conda/envs/glm52/bin/python}"
NLLB_CALIBRATION_SEED="${NLLB_CALIBRATION_SEED:-812701}"
output_path="artifacts/diagnostics/tatoeba_policy_calibration_${SLURM_JOB_ID}.json"

"$NLLB_PYTHON" -u src/calibrate_tatoeba_policy.py \
  --data data/tatoeba_en_de/protocol_v1/calibration.jsonl \
  --output "$output_path" \
  --seed "$NLLB_CALIBRATION_SEED"
