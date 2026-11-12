# Predicting reward overoptimization in machine translation

Research code, frozen protocols, and paper sources for studying what happens
when machine translation policies optimize learned quality estimation rewards.
The current question is whether a cheap audit of a reward against an initial
policy can predict which failure mode reinforcement learning will amplify.

The repository grew from an earlier tokenization intervention study. That
branch is closed as a bounded negative result and has its own short paper. The
main project now combines controlled candidate selection experiments with
generative GRPO experiments.

## Project status

### Reward overoptimization

The completed controlled studies include:

- best-of-$N$ selection with fixed WMT candidate pools;
- a finite-action contextual bandit whose expected objective and gradient can
  be computed exactly;
- a source-and-candidate-text policy evaluated on held-out systems;
- reward-training seed and optimization-horizon replications;
- prospective tests of pre-optimization screening and pressure selection; and
- matched controls for metric--annotation disagreement filtering.

These studies support the class project and the current ACL draft. See the
[reward-overoptimization protocol](docs/REWARD_OVEROPTIMIZATION_PROTOCOL.md),
[contextual-bandit protocol](docs/CONTEXTUAL_BANDIT_PROTOCOL.md), and
[pre-optimization predictor protocol](docs/PREOPTIMIZATION_PREDICTOR_PROTOCOL.md).

### Generative GRPO

The active extension trains Qwen translation policies with COMET-family and
MetricX quality-estimation rewards. A policy-relative audit scores controlled
source-copy, refusal, repetition, and shortening behaviors against the initial
policy before training. Its prospective calls and abstentions are frozen in
[`results/grpo_scale_predictions.json`](results/grpo_scale_predictions.json).

The scale experiment is defined by
[`results/grpo_scale_protocol.json`](results/grpo_scale_protocol.json). The
left-padding implementation correction, launch provenance, and planned result
scoring are recorded in:

- [`results/grpo_scale_protocol_amendment_1.json`](results/grpo_scale_protocol_amendment_1.json)
- [`results/grpo_scale_launch.json`](results/grpo_scale_launch.json)
- [`src/analyze_grpo_scale.py`](src/analyze_grpo_scale.py)
- [`docs/GRPO_PAPER_PLAN.md`](docs/GRPO_PAPER_PLAN.md)

Scale outcomes are added to the paper only after every expected run artifact
passes the scorer's provenance, row-trace, indicator, and manipulation checks.

### Tokenization study

The earlier branch established exact same-string policy-gradient invariance
and closed a token-length-tilt intervention after its launch screen did not
pass. It does not claim a translation-learning effect. See
[`docs/SAME_STRING_INVARIANCE.md`](docs/SAME_STRING_INVARIANCE.md),
[`docs/GO_NO_GO.md`](docs/GO_NO_GO.md), and
[`paper/tokenization_acl/`](paper/tokenization_acl/).

## Papers and class deliverables

- [`paper/reward_overoptimization_acl/`](paper/reward_overoptimization_acl/):
  ACL-format manuscript for the reward-overoptimization project.
- [`paper/tokenization_acl/`](paper/tokenization_acl/): short negative-result
  paper for the tokenization branch.
- [`paper/class_project_proposal/`](paper/class_project_proposal/): proposal for
  the reinforcement-learning course project.

The reward-overoptimization manuscript currently presents the controlled
candidate studies. The generative GRPO section will be integrated after the
frozen scale analysis is complete.

## Reproducibility and numeric claims

[`results/paper_stats.json`](results/paper_stats.json) is the single source of
truth for paper-facing experimental quantities. Analysis scripts merge their
own keys into that file. Paper builders render LaTeX commands into
`generated_stats.tex`; those generated files are build artifacts and are not
edited by hand.

Before considering a paper build complete:

1. Recompute the relevant analysis section in `paper_stats.json`.
2. Regenerate the paper's LaTeX commands and figures.
3. Run [`scripts/check_tex_numbers.py`](scripts/check_tex_numbers.py) on every
   hand-authored TeX source.
4. Confirm that each reported table row traces to a source record or run
   artifact.

Build the main manuscript from the repository root:

```bash
python src/build_reward_overoptimization_paper.py
python src/plot_reward_overoptimization.py
python src/plot_contextual_bandit.py \
  --output-stem paper/reward_overoptimization_acl/figures/contextual_bandit
python src/plot_text_policy_bandit.py \
  --output-stem paper/reward_overoptimization_acl/figures/text_policy_bandit
python scripts/check_tex_numbers.py paper/reward_overoptimization_acl/main.tex
make -C paper/reward_overoptimization_acl
```

Build the tokenization paper:

```bash
python src/build_tokenization_paper_stats.py
python scripts/check_tex_numbers.py paper/tokenization_acl/main.tex
make -C paper/tokenization_acl
```

The paper directories also provide a `tectonic` Make target.

## Repository layout

- `src/`: data preparation, reward training, audits, policy optimization,
  evaluation, analysis, paper-stat generation, and plotting.
- `results/`: frozen protocols, prospective predictions, compact audit output,
  and the paper statistics source of truth.
- `docs/`: scientific protocols, intervention decisions, human-evaluation
  guidance, and research plans.
- `scripts/slurm/`: tracked BYU cluster launchers for the experiments.
- `paper/`: ACL manuscripts and the course proposal.
- `tests/`: checks for the controlled data, reward, bandit, and analysis paths.
- `artifacts/`: local generated translations, logs, checkpoints, and evaluation
  output. Most contents are intentionally excluded from Git.
- `data/`: compact provenance and split audits. Corpus text remains local.

## Main execution paths

### Controlled reward study

The main sequence is:

1. inventory and diagnose DA slices;
2. build matched reward-training conditions;
3. train and score reward models;
4. run best-of-$N$, contextual-bandit, and text-policy analyses;
5. run the frozen prospective confirmation and pressure-selection analyses;
6. merge results into `paper_stats.json` and rebuild the paper.

The corresponding scripts use descriptive names under `src/`, including
`analyze_da_slice_diagnostics.py`, `train_da_reward_model.py`,
`analyze_wmt25_best_of_n.py`, `analyze_wmt26_best_of_n.py`,
`run_contextual_bandit.py`, `run_text_policy_bandit.py`, and the predictor and
safe-pressure analysis scripts.

### Generative GRPO study

The generative path is centered on:

- [`src/audit_policy_relative.py`](src/audit_policy_relative.py)
- [`src/train_grpo_pilot.py`](src/train_grpo_pilot.py)
- [`src/eval_grpo_pilot.py`](src/eval_grpo_pilot.py)
- [`src/analyze_grpo_rule_tests.py`](src/analyze_grpo_rule_tests.py)
- [`src/analyze_grpo_scale.py`](src/analyze_grpo_scale.py)

Cluster launchers default to offline Hugging Face caches and the local GRPO
Python environment. Do not regenerate a frozen protocol or prediction file in
place. Each freezing script refuses to overwrite its output, and trainers pin
the expected hashes.

## Environment

There is no single environment file covering every historical branch.
[`requirements-rl.txt`](requirements-rl.txt) and
[`requirements-data.txt`](requirements-data.txt) contain the small pinned
dependencies for the original NLLB and data-reader paths. The GRPO launchers
use the cluster's dedicated `~/.venvs/grpo` environment together with local
model caches. Protocol JSON files pin model identifiers, input hashes, and
experimental settings.

The repository is designed for the BYU Slurm cluster, but the analysis and
paper-building scripts can run elsewhere when their recorded inputs are
available. Scheduler placement is operational metadata and is not part of the
scientific treatment.

## Data, models, and redistribution

WMT annotations and system outputs, Tatoeba text, model checkpoints, generated
translations, and optimizer state are not distributed in this repository.
Compact source hashes, manifests, and attribution records are retained where
permitted. The NLLB/Tatoeba branch's terms and attribution are summarized in
[`docs/DATA_MODEL_NOTICE.md`](docs/DATA_MODEL_NOTICE.md). Consult the relevant
WMT release and model cards before redistributing data or modified weights.

OPUS-100 remains excluded from research training because its combined-source
licensing was not cleared for this project.

No code license has been assigned to this repository.
