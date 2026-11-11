#!/bin/bash
#SBATCH --job-name=RL_scale
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=artifacts/logs/RL_scale_%A_%a.out
# Frozen scale protocol (results/grpo_scale_protocol.json); array index = sorted run id.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENSSL_CONF=/dev/null
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
read -r run protocol reward seed <<< "$(python3 -c "
import json,sys; r=json.load(open('results/grpo_scale_protocol.json'))['runs']; k=sorted(r)[int(sys.argv[1])]
print(k, r[k]['protocol'], r[k]['reward'], r[k]['seed'])" "$SLURM_ARRAY_TASK_ID")"
echo "run=$run"
out="/nobackup/archive/usr/it238/reinforcement-learning/grpo_scale/${run}"
keep="$PWD/artifacts/grpo_scale/${run}"
cd src
"$PY" -u train_grpo_pilot.py --scale-run "$run" --protocol-name "$protocol" --reward "$reward" --seed "$seed" --output-dir "$out"
"$PY" -u eval_grpo_pilot.py --protocol-name "$protocol" --run-dir "$out"
mkdir -p "$keep" && cp "$out"/eval_summary.json "$out"/eval_outputs.jsonl "$out"/pilot_run.json "$keep"/
