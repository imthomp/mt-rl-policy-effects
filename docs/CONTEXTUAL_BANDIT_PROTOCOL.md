# Contextual-bandit class study

## Question and claim boundary

The study asks how policy-gradient optimization of particular learned MT
rewards changes proxy reward and human-rated translation quality in a fixed
candidate environment. It does not treat superiority of metric--annotation
disagreement filtering as the expected result and does not test a new RL
algorithm.

The main environment is the frozen WMT26 Simplified-Chinese-to-Japanese cESA
pool. A context is one source segment and an action is one released system
translation for that segment. The policy cannot generate an action outside the
released candidate pool.

## Frozen split and features

Documents, and therefore all of their source segments, are assigned to train
or test before policy fitting. Assignment is deterministic, stratified by
domain, and uses a seventy--thirty train--test split. No human score enters the
split.

The shared contextual policy uses a fixed signed hashing representation of
source-character three- through five-grams, concatenated with domain
indicators. It learns a linear logit for every candidate system. The hashing
transform has no fitted vocabulary and therefore cannot leak test sources into
training. A separate tabular policy over training contexts demonstrates
known-source memorization and is never described as source generalization.

## Objective and optimization

For each frozen reward checkpoint, the training reward is standardized using
only reward scores from training contexts. Human cESA is stored in a separate
evaluation matrix and is never passed to an optimizer, baseline, stopping
rule, checkpoint-selection rule, or hyperparameter decision.

Both the exact policy-gradient reference and sampled REINFORCE maximize the
same mean expected standardized proxy reward from identical zero-logit
initialization. They use the same full set of training contexts, learning
rate, update count, and no entropy bonus. REINFORCE samples one action per
training context per update. Its zero baseline is valid because the training
reward has been centered on the training pool.

Five fixed policy-training seeds quantify policy-training variability only.
There is one reward-model checkpoint per reward condition, so the experiment
does not estimate reward-model-training variability.

The fixed shared-policy representation has 512 hashing features plus domain
indicators. Both optimizers use full-context stochastic-gradient descent with
a learning rate of 0.5 for 200 updates; measurements are recorded every 10
updates. The REINFORCE seeds are 20261110 through 20261114. These choices may
be changed only to correct an execution failure, with a new protocol version,
not in response to human-quality results.

## Exact-gradient audit

At fixed policy checkpoints from the exact trajectory, compute the exact
gradient by enumerating every action. At each frozen checkpoint, repeatedly
resample the one-action-per-context REINFORCE gradient without updating the
policy. Report the mean-gradient bias norm, gradient-estimator variance, and
cosine agreement with the exact gradient. Policy drift cannot enter this
comparison.

The frozen exact-trajectory checkpoints are updates 0, 50, 100, and 200. Each
gradient audit uses 256 independent estimator draws without a parameter
update.

## Primary outcomes

For exact optimization and every REINFORCE seed, record expected human cESA on
train and source-disjoint test contexts against:

1. training update;
2. achieved standardized proxy reward, expressed as gain over uniform
   selection in training-reward standard-deviation units.

Raw proxy reward is retained for condition-specific interpretation but is not
used as a shared horizontal scale across reward checkpoints.

The fixed evaluation baselines are uniform selection, direct proxy-best
selection, and the human-best candidate ceiling. Human-best is an evaluation
ceiling only and is never available to the learned policy.

## Interpretation

Increasing assigned reward is a positive policy-learning and engineering
result. It is not evidence of improved translation quality. Source-disjoint
performance measures generalization of this fixed candidate-selection policy,
not generalization of a generative translation model. WMT25 and WMT26 informed
the project framing and are not described as untouched confirmation of the
broader filtering hypothesis.
