#!/bin/bash
#SBATCH --job-name=RL_scale_smoke
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/logs/RL_scale_smoke_%j.out
# Two-step implementation check for the first frozen scale run; never analyzed as a result.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENSSL_CONF=/dev/null
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
read -r run protocol reward seed <<< "$(python3 -c "
import json; r=json.load(open('results/grpo_scale_protocol.json'))['runs']; k=sorted(r)[0]
print(k, r[k]['protocol'], r[k]['reward'], r[k]['seed'])")"
echo "run=$run"
cd src
"$PY" -u train_grpo_pilot.py --smoke --scale-run "$run" --protocol-name "$protocol" --reward "$reward" --seed "$seed" --output-dir /tmp/ignored
