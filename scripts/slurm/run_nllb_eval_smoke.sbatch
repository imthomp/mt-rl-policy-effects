#!/bin/bash
#SBATCH --job-name=nllb_eval_smoke
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/logs/nllb_eval_smoke_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLLB_ALLOCATOR_LIMIT_GIB="${NLLB_ALLOCATOR_LIMIT_GIB:-72}"
export NLLB_MODEL_DIR="${NLLB_MODEL_DIR:-${HOME}/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d}"
NLLB_PYTHON="${NLLB_PYTHON:-${HOME}/.conda/envs/glm52/bin/python}"
smoke_dir="artifacts/diagnostics/eval_smoke_${SLURM_JOB_ID}"
mkdir -p "$smoke_dir"

"$NLLB_PYTHON" -u src/evaluate_nllb_policy.py \
  --data data/tatoeba_en_de/protocol_v1/development.jsonl \
  --output-dir "$smoke_dir/evaluations" \
  --max-examples 8 \
  --checkpoint control=base \
  --checkpoint treatment=base

"$NLLB_PYTHON" -u src/summarize_nllb_policy_evaluations.py \
  --metrics "control=$smoke_dir/evaluations/metrics_control.json" \
  --metrics "treatment=$smoke_dir/evaluations/metrics_treatment.json" \
  --paired-seed 17:control:treatment \
  --bootstrap-replicates 100 \
  --output "$smoke_dir/paired_summary.json"
