#!/bin/bash
#SBATCH --job-name=mt_text_bandit
#SBATCH --partition=m9g
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:1
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/logs/text_policy_bandit_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TRAINING_PYTHON="${TRAINING_PYTHON:-${HOME}/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/.venv/bin/python}"

"$TRAINING_PYTHON" -u src/run_text_policy_bandit.py \
  --prediction-root artifacts/reward_models/wmt26_predictions \
  --campaign-archive data/reward_da/wmt26-campaign_sources.zip \
  --embedding-cache artifacts/bandit/text_pair_embeddings.npz \
  --probabilities-output artifacts/bandit/text_policy_heldout_probabilities.npz \
  --output results/paper_stats.json
