#!/bin/bash
#SBATCH --job-name=RL_grpo_rule
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-6
#SBATCH --output=artifacts/logs/grpo_rule_%A_%a.out

# Runs that test results/grpo_audit_rule.json:
#   0-2 MetricX-24-QE en-es seeds 0-2 (pilot protocol); 3-6 en-de seed 0, four rewards;
#   7-14 en-de seeds 1-2 (added 2026-11-07 after seed-0 refusal collapse).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENSSL_CONF=/dev/null
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
runs=(
  "pilot metricx_24_qe 0" "pilot metricx_24_qe 1" "pilot metricx_24_qe 2"
  "ende xcomet_xl_qe 0" "ende cometkiwi_22 0" "ende cometkiwi_23_xl 0" "ende metricx_24_qe 0"
  "ende xcomet_xl_qe 1" "ende cometkiwi_22 1" "ende cometkiwi_23_xl 1" "ende metricx_24_qe 1"
  "ende xcomet_xl_qe 2" "ende cometkiwi_22 2" "ende cometkiwi_23_xl 2" "ende metricx_24_qe 2"
)
read -r protocol reward seed <<< "${runs[$SLURM_ARRAY_TASK_ID]}"
if [[ "$protocol" == pilot ]]; then lp=en_es; else lp=en_de; fi
# Checkpoints (~6 GB each) go to /nobackup; home quota is 2 TB. Small eval files are copied back.
out="/nobackup/archive/usr/it238/reinforcement-learning/grpo_rule_tests/${lp}/${reward}_seed${seed}"
keep="$PWD/artifacts/grpo_rule_tests/${lp}/${reward}_seed${seed}"
cd src
"$PY" -u train_grpo_pilot.py --protocol-name "$protocol" --reward "$reward" --seed "$seed" --output-dir "$out"
"$PY" -u eval_grpo_pilot.py --protocol-name "$protocol" --run-dir "$out"
mkdir -p "$keep" && cp "$out"/eval_summary.json "$out"/eval_outputs.jsonl "$out"/pilot_run.json "$keep"/
