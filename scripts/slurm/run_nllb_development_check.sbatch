#!/bin/bash
#SBATCH --job-name=nllb_dev_check
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/logs/nllb_dev_check_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLLB_ALLOCATOR_LIMIT_GIB="${NLLB_ALLOCATOR_LIMIT_GIB:-72}"
export NLLB_MODEL_DIR="${NLLB_MODEL_DIR:-${HOME}/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d}"
NLLB_PYTHON="${NLLB_PYTHON:-${HOME}/.conda/envs/glm52/bin/python}"
control_dir="artifacts/runs/development_check_${SLURM_JOB_ID}_control"
treatment_dir="artifacts/runs/development_check_${SLURM_JOB_ID}_treatment"
evaluation_dir="artifacts/runs/development_check_${SLURM_JOB_ID}_evaluation"

"$NLLB_PYTHON" -u src/nllb_policy_rl.py \
  --data data/tatoeba_en_de/protocol_v1/train.jsonl \
  --output "$control_dir" \
  --beta 0.0 \
  --run-seed 17 \
  --subset-seed 314159 \
  --development-check

"$NLLB_PYTHON" -u src/nllb_policy_rl.py \
  --data data/tatoeba_en_de/protocol_v1/train.jsonl \
  --output "$treatment_dir" \
  --beta 0.20 \
  --run-seed 17 \
  --subset-seed 314159 \
  --development-check

"$NLLB_PYTHON" -u src/evaluate_nllb_policy.py \
  --data data/tatoeba_en_de/protocol_v1/development.jsonl \
  --output-dir "$evaluation_dir" \
  --checkpoint base=base \
  --checkpoint "control=${control_dir}/final.pt" \
  --checkpoint "treatment=${treatment_dir}/final.pt" \
  --max-examples 8

"$NLLB_PYTHON" -u src/summarize_nllb_policy_evaluations.py \
  --metrics "base=${evaluation_dir}/metrics_base.json" \
  --metrics "control=${evaluation_dir}/metrics_control.json" \
  --metrics "treatment=${evaluation_dir}/metrics_treatment.json" \
  --paired-seed 17:control:treatment \
  --output "${evaluation_dir}/paired_summary.json" \
  --bootstrap-replicates 100
