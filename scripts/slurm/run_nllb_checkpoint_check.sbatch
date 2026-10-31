#!/bin/bash
#SBATCH --job-name=nllb_ckpt_check
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/logs/nllb_ckpt_check_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLLB_ALLOCATOR_LIMIT_GIB="${NLLB_ALLOCATOR_LIMIT_GIB:-72}"
export NLLB_MODEL_DIR="${NLLB_MODEL_DIR:-${HOME}/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d}"
NLLB_PYTHON="${NLLB_PYTHON:-${HOME}/.conda/envs/glm52/bin/python}"
run_dir="artifacts/runs/checkpoint_check_${SLURM_JOB_ID}"

"$NLLB_PYTHON" -u src/nllb_policy_rl.py \
  --data data/tatoeba_en_de/protocol_v1/train.jsonl \
  --output "$run_dir" \
  --beta 0.0 \
  --run-seed 17 \
  --subset-seed 314159 \
  --checkpoint-check

"$NLLB_PYTHON" -u src/verify_nllb_checkpoint.py \
  --model-dir "$NLLB_MODEL_DIR" \
  --run-dir "$run_dir"
