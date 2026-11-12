#!/bin/bash
#SBATCH --job-name=RL_v2_rerun
#SBATCH --partition=dw,dwmatrix
#SBATCH --qos=matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --array=0-44
#SBATCH --output=artifacts/logs/RL_v2_rerun_%A_%a.out
# Left-padding reruns of the 1.5B grid (results/grpo_padding_rerun_amendment.json); array index = run index.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENSSL_CONF=/dev/null
PY="${GRPO_PYTHON:-${HOME}/.venvs/grpo/bin/python}"
eval "$(python3 - "$SLURM_ARRAY_TASK_ID" <<'PYEOF'
import json, shlex, sys
r = json.load(open("results/grpo_padding_rerun_amendment.json"))["rerun"]["runs"][int(sys.argv[1])]
name = r["reward"] + ("_guarded" if r["guarded"] else "") + ("_langid" if r["langid"] else "") \
       + (f"_steps{r['steps']}" if r["steps"] else "") + (f"_seed{r['seed']}" if r["seed"] is not None else "")
train = ["--protocol-name", r["protocol"], "--reward", r["reward"]]
if r["seed"] is not None: train += ["--seed", str(r["seed"])]
if r["guarded"]: train += ["--guarded"]
if r["steps"]: train += ["--steps", str(r["steps"])]
if r["langid"]: train += ["--langid-guard"]
ev = ["--protocol-name", r["protocol"]] + (["--steps", str(r["steps"])] if r["steps"] else [])
print(f"group={shlex.quote(r['group'])}; name={shlex.quote(name)}; "
      f"train=({' '.join(map(shlex.quote, train))}); ev=({' '.join(map(shlex.quote, ev))})")
PYEOF
)"
echo "group=$group name=$name"
out="/nobackup/archive/usr/it238/reinforcement-learning/grpo_v2/${group}/${name}"
keep="$PWD/artifacts/grpo_v2/${group}/${name}"
cd src
"$PY" -u train_grpo_pilot.py "${train[@]}" --output-dir "$out"
"$PY" -u eval_grpo_pilot.py "${ev[@]}" --run-dir "$out"
mkdir -p "$keep" && cp "$out"/eval_summary.json "$out"/eval_outputs.jsonl "$out"/pilot_run.json "$keep"/
