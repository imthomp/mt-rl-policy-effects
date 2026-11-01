# Prospective pre-optimization failure predictor

Status: design draft. Do not run a confirmation analysis from this document
until the hypothesis, human-label budget, untouched datasets, predictor formulas,
and decision thresholds are frozen in a dated revision.

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

Development and confirmation must be disjoint in reward seed, source or
document, and held-out distribution. Complete systems should also be held out
where candidate coverage permits it.

## Outcome

The primary outcome is change in independent human quality from the initial
policy at matched optimization pressure. Pressure will be expressed using a
frozen policy-movement measure such as KL from the initial policy or policy
entropy, not update count. This prevents a slowly moving policy from being
misclassified as safe.

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

## Predictor requiring independent human labels

### First-order alignment

On a separately labeled calibration subset, compute the dot product between the
reward objective's initial policy gradient and the human-quality objective's
initial gradient. A negative value predicts immediate movement against human
quality. This predictor consumes human labels and must be compared at matched
total cost, not presented as free.

Labels used for this predictor may not be reused for confirmation outcomes.

## Baselines

- Development correlation on the calibration pool.
- A small best-of-N audit on identical candidate pools.
- Initial reward--human agreement within source.
- Always keep the initial policy.
- Always optimize the reward.

The best-of-N audit is the primary practical baseline. Its sampling budget and
human-label cost must match the proposed screen as closely as possible.

## Frozen decision rule

The development data may be used once to choose one predictor specification and
one decision threshold. The rule must map each reward checkpoint to either
`optimize` or `keep_initial` without consulting confirmation outcomes.

Before confirmation, freeze:

- predictor formulas, standardization, and missing-data handling;
- the policy-movement grid and quality-loss tolerance;
- the decision threshold and tie handling;
- label, scoring, and compute budgets;
- reward seeds and source/system/distribution splits;
- primary uncertainty interval and success criterion;
- handling of failed reward training or optimization runs.

No predictor may be added after inspecting confirmation outcomes.

## Confirmation requirements

Confirmation requires genuinely new human-rated candidate data. Repartitioning
the already inspected WMT pools does not restore untouched status. If suitable
released data do not exist, the choices are to collect a bounded new human
evaluation or explicitly keep the work as an exploratory development study.

The confirmation report must show every reward checkpoint, including failed
or degenerate runs, and intervals over reward-training units. It must report
selection quality, regret, worst-case loss, abstention rate, and total cost.

## Interpretation boundaries

A successful system-level predictor in a fixed candidate pool may exploit
stable system style. It does not by itself establish a safeguard for generative
RL, where system identity is absent and the policy creates new outputs.

A successful first-order predictor establishes local alignment near the initial
policy. It does not guarantee safety under large policy movement unless the
confirmation outcome explicitly tests that range.

## Unresolved choices before freezing

- The untouched human-rated datasets and their candidate coverage.
- The human-label budget for first-order alignment.
- The matched policy-pressure grid.
- The practical human-quality loss tolerance.
- Whether confirmation targets fixed-pool selection only or also a generative
  policy experiment.
