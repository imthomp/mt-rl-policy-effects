#!/bin/bash
#SBATCH --job-name=RL_gen
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=artifacts/logs/RL_gen_%A_%a.out
# Frozen generalization protocol (results/grpo_generalization_protocol.json); array index = sorted run id.
# Arm C (indices 0-11) needs --gres=gpu:2; arm D (12-35) one GPU. Submit them separately.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENSSL_CONF=/dev/null
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
read -r run protocol reward seed gpus <<< "$(python3 -c "
import json,sys; r=json.load(open('results/grpo_generalization_protocol.json'))['runs']; k=sorted(r)[int(sys.argv[1])]
print(k, r[k]['protocol'], r[k]['reward'], r[k]['seed'], r[k].get('gpus', 1))" "$SLURM_ARRAY_TASK_ID")"
echo "run=$run gpus=$gpus visible=${CUDA_VISIBLE_DEVICES:-none}"
ngpu=$(nvidia-smi -L | wc -l)
if [[ "$gpus" != "$ngpu" ]]; then echo "protocol wants $gpus GPU(s) but job has $ngpu" >&2; exit 1; fi
if [[ "$gpus" == 2 ]]; then export REWARD_DEVICE=1; fi
out="/nobackup/archive/usr/it238/reinforcement-learning/grpo_gen/${run}"
keep="$PWD/artifacts/grpo_gen/${run}"
cd src
"$PY" -u train_grpo_pilot.py --gen-run "$run" --protocol-name "$protocol" --reward "$reward" --seed "$seed" --output-dir "$out"
"$PY" -u eval_grpo_pilot.py --protocol-name "$protocol" --run-dir "$out"
mkdir -p "$keep" && cp "$out"/eval_summary.json "$out"/eval_outputs.jsonl "$out"/pilot_run.json "$keep"/
