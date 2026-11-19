# Predicting reward overoptimization in machine translation

Research code, frozen protocols, and paper sources for studying what happens
when machine translation policies optimize learned quality estimation rewards.
The current question is whether a cheap audit of a reward against an initial
policy can predict which failure mode reinforcement learning will amplify.

**Resuming work?** Start at [`notes/HANDOFF.md`](notes/HANDOFF.md) (its CURRENT STATE block is kept up to date). The day-by-day history is in [`notes/research_log.md`](notes/research_log.md).

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

Qwen2.5 translation policies (1.5B full fine-tuning; 7B LoRA) are trained with GRPO against COMET-family and MetricX quality-estimation rewards.

- **The audit:** before training, a policy-relative audit compares each reward's scores for controlled source-copy, refusal, repetition and truncation outputs against the initial policy's own translations.
- **The rule:** a frozen three-band rule ([`results/policy_relative_rule.json`](results/policy_relative_rule.json)) turns each preference rate into a hack call, a safe call, or an abstention.
- **Freezing:** every study's calls are frozen before training, and the trainer refuses runs whose protocol or prediction hashes differ from the pinned values.

Studies, all complete unless noted:

| Study | Protocol / predictions | Result |
|---|---|---|
| Corrected 1.5B grid (en-es, en-de, en-zh) | [`grpo_padding_rerun_amendment.json`](results/grpo_padding_rerun_amendment.json) | Reran every early run after a rollout right-padding bug; the earlier refusal collapse and preambles were artifacts of it |
| Scale study (7B LoRA; MetricX-24 XL/XXL rewards) | [`grpo_scale_protocol.json`](results/grpo_scale_protocol.json), [`grpo_scale_predictions.json`](results/grpo_scale_predictions.json) | Safe calls 36/36 tasks; hack calls 1/2 (XCOMET copies into Chinese, not German) |
| Generalization study (XCOMET-XXL, CometKiwi-23-XXL; en-ja, en-ru) | [`grpo_generalization_protocol.json`](results/grpo_generalization_protocol.json), [`grpo_generalization_predictions.json`](results/grpo_generalization_predictions.json) | Hack calls 4/4 tasks (12/12 cells); safe calls 27/34 tasks, every failure in the truncation indicator and mostly cleanups of bloated initial outputs |
| Reachability probe (why 7B en-de did not copy) | [`reachability_probe_protocol.json`](results/reachability_probe_protocol.json) | Failed calibration, frozen before the generalization outcomes; how often the policy samples copies does not explain it |
| Copy-margin hypothesis (post hoc, disclosed) | [`copy_margin_protocol.json`](results/copy_margin_protocol.json), [`copy_margin_predictions.json`](results/copy_margin_predictions.json) | Frozen before the generalization outcomes; consistent with all four new hack tasks, including en-ru |
| **Fresh copy-margin test (in progress)** | [`margin_test_protocol.json`](results/margin_test_protocol.json) | Protocol frozen; candidate audits complete (`results/policy_relative_audit_margin_M_*.json`). Runs and predictions NOT yet frozen ([`src/freeze_margin_test_runs.py`](src/freeze_margin_test_runs.py)) or launched |

Reference quality (COMET-22, chrF vs WMT24 refA) agrees with the audit: runs without a realized hack improve, and copying runs lose heavily. Blinded human evaluation is prepared but not yet sent. Primary sheets cover the 7B safe-called rewards and the MetricX large/XL/XXL series:
- annotators receive only their `sheet.csv` and [`docs/HUMAN_EVAL_ANNOTATOR_GUIDE.md`](docs/HUMAN_EVAL_ANNOTATOR_GUIDE.md);
- the study team's notes and planned analysis are in [`docs/HUMAN_EVAL_GUIDELINES.md`](docs/HUMAN_EVAL_GUIDELINES.md).

The paper outline and framing decisions are in [`docs/GRPO_PAPER_OUTLINE.md`](docs/GRPO_PAPER_OUTLINE.md). Prior-work positioning, with every citation checked against its source, is in [`notes/lit_review_2026-11-11.md`](notes/lit_review_2026-11-11.md).

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
- [`paper/grpo_audit_acl/`](paper/grpo_audit_acl/): ACL-format manuscript for
  the prospective generative reward-audit study. It covers the scale and generalization studies, the audit-outcome figure, and the padding correction. Human evaluation is pending, and the abstract and outline are drafts for the author to revise.
- [`paper/tokenization_acl/`](paper/tokenization_acl/): short negative-result
  paper for the tokenization branch.
- [`paper/class_project_proposal/`](paper/class_project_proposal/): proposal for
  the reinforcement-learning course project.

The reward-overoptimization manuscript presents the controlled candidate
studies. The generative GRPO study is maintained as a separate manuscript so
its prospective design and padding correction remain explicit.

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

Build the generative GRPO audit paper (order matters: the builder refuses a stale figure):

```bash
module load texlive/2024
python src/analyze_grpo_generalization.py          # only after new generalization outputs
python src/diagnose_truncation_flags.py
~/.venvs/grpo/bin/python src/plot_audit_outcomes.py
python src/build_grpo_audit_paper.py
python scripts/check_tex_numbers.py paper/grpo_audit_acl/main.tex
make -C paper/grpo_audit_acl
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
- [`src/analyze_grpo_ende_followup.py`](src/analyze_grpo_ende_followup.py)
- [`src/build_grpo_audit_paper.py`](src/build_grpo_audit_paper.py)

Supporting scripts:

| Role | Scripts |
|---|---|
| Freezing (each refuses to overwrite its output) | `freeze_grpo_*_protocol.py`, `freeze_grpo_*_predictions.py`, `freeze_reachability_predictions.py`, `freeze_margin_predictions.py`, `freeze_margin_test_runs.py` |
| Scoring | `analyze_grpo_generalization.py`, plus the scale and rule-test scorers above |
| Probes and diagnostics | `probe_reachability.py`, `diagnose_truncation_flags.py`, `audit_preamble_exploratory.py` |
| Reference quality | `eval_grpo_reference.py` (COMET-22, chrF, paired bootstrap; cached), `eval_stripped_reference.py` |
| Figure | `plot_audit_outcomes.py` (task-level scatter; palette validated for colour vision deficiency) |
| Human evaluation | `prepare_human_eval.py` (blinded sheets; key hashes in `artifacts/human_eval/manifest.json`) |

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

`~/.venvs/grpo` (torch 2.6 + cu124, TRL 0.25.1, COMET) also has these additions:
- `peft` (LoRA);
- `fasttext-wheel` (the GlotLID language-ID guard; model in the Hugging Face cache);
- `matplotlib` (figures).

NumPy stays pinned at 1.26.4. The larger reward models (XCOMET-XXL, CometKiwi-23-XXL, MetricX-24 XL and XXL) live in `/nobackup/archive/usr/it238/hf_cache` because of the home quota. Runs with an XXL reward use 2 GPUs: the policy on GPU 0 and the reward on GPU 1 (`REWARD_DEVICE=1`). torch 2.6 here does not support B200 GPUs.

Slurm conventions:
- job names are `RL_<desc>`;
- placement comes from the lab's [byurc](https://github.com/byu-matrix-lab/byurc) (`~/byurc/bin/byurc pick-partition --eta ...`); move pending jobs with `requeue_to_available.sh`;
- set time limits from measured runtimes.

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
