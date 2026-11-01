# Prospective pre-optimization failure predictor

Status: frozen for confirmation. The machine-readable source of truth is
`results/text_policy_predictor_protocol.json`; the exact candidate, document,
and system lists are in `results/predictor_confirmation_manifest.json`. Human
outcomes for the confirmation evaluation documents remained sealed when this
revision was frozen.

The existing WMT candidate pools, reward checkpoints, and observed optimization
outcomes are development evidence. They cannot serve as untouched confirmation
data for this study.

## Research question

Can quantities measured before policy optimization predict whether optimizing a
machine-translation reward will reduce independent human quality, well enough
to support the practical choice between optimizing that reward and retaining
the initial policy?

The intended contribution is a cheap screening decision, not another claim
that generic reward overoptimization exists.

## Unit of analysis

The unit is an independently trained reward checkpoint evaluated on a frozen
source--candidate pool. Reward-training seeds are independent units. Policy
sampling seeds estimate optimizer noise and are not substitutes for reward
seeds.

Development and confirmation are disjoint in reward seed and dataset.
Calibration and evaluation are document-disjoint within each confirmation
dataset. The source--candidate-text policy is trained and evaluated on disjoint
system sets and never receives system identity.

## Outcome

The primary outcome is change in independent human quality from the uniform
policy at the frozen primary KL target. Secondary outcomes trace the complete
frozen KL grid. Checkpoints are selected by achieved KL, never by human quality
or update count.

Secondary outcomes are the human-quality curve against achieved proxy reward,
the pressure at which quality first falls below its frozen tolerance, and
regret of the screening decision relative to the best available choice.

The decision set must include `keep_initial`. A screen is not useful if it is
forced to select a reward that it predicts will fail.

## Predictors available without new human labels

All reward scores are standardized using a frozen calibration pool before these
predictors are computed.

### Top-system preference margin

For each system, average the standardized reward across calibration sources.
The predictor is the gap between the largest system mean and the grand mean.
It measures how strongly the reward favors one system before optimization.

### Between-system separation

Fit a source-and-system decomposition to standardized reward scores. The
predictor is the prespecified fraction of residual score variance attributable
to system identity. It measures broad system separation rather than only the
single most favored system.

### Initial policy movement

At the uniform policy, calculate the exact reward-gradient norm and the
first-order change in entropy or KL under the frozen update rule. This separates
weak learning signals from rewards that immediately induce concentrated
policies.

## Frozen primary predictor

### First-order alignment

On the frozen document-disjoint calibration subset, compute the cosine between
the reward objective's exact initial gradient and the human-quality objective's
exact initial gradient in the shared source--candidate-text scorer. The reward
gradient uses the full unlabeled optimization pool; only the human gradient
uses the bounded labeled sample.

The decision follows the theoretically defined sign: optimize only when the
estimated alignment is strictly positive. A tie, nonfinite value, or negative
alignment keeps the initial policy. The selected label budget and development
stability diagnostics are recorded in the protocol JSON and SSOT.

Labels used for this predictor may not be reused for confirmation outcomes.

## Baselines

- Development correlation on the calibration pool.
- A small best-of-N audit on identical candidate pools.
- Initial reward--human agreement within source.
- Always keep the initial policy.
- Always optimize the reward.

The best-of-N audit is the primary practical baseline. The frozen protocol uses
exact expectations over candidate subsets and the same calibration outcomes.

## Frozen decision and analysis

The formulas, standardization, label budget, reward seeds, policy-pressure grid,
system and document splits, baselines, tie handling, and success criterion are
now frozen in the protocol JSON. Every failed or degenerate run remains in the
report. No predictor may be added after inspecting confirmation outcomes.

## Confirmation requirements

The primary and secondary confirmation pools are the frozen WMT MQM datasets
listed in the manifest. Their existence had been mentioned prospectively, so
they are not historically unanticipated, but their MQM outcomes were not used
in predictor development. Calibration outcomes may be opened only to compute
the frozen screen; evaluation-document outcomes remain sealed until decisions
and policy trajectories are fixed.

The confirmation report must show every reward checkpoint, including failed
or degenerate runs, and intervals over reward-training units. It must report
selection quality, regret, worst-case loss, abstention rate, and total cost.

## Interpretation boundaries

A successful predictor in this fixed candidate pool does not by itself
establish a safeguard for generative RL, where the policy creates new outputs.

A successful first-order predictor establishes local alignment near the initial
policy. It does not guarantee safety under large policy movement unless the
confirmation outcome explicitly tests that range.

## Scope

This confirmation targets fixed-pool source--candidate-text selection. A
generative policy experiment is explicitly outside the frozen study. The
remaining work is execution: train the new reward seeds, score the frozen
candidate pools, encode their text pairs, apply the calibration-only screen,
run every policy trajectory, and then evaluate on the sealed documents.
