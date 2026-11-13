# Generative GRPO paper: current state and completion plan

Status: the corrected training grid, prospective scale study, frozen
scorecards, reference evaluation, stripped-output analysis, and preamble audit
are complete. The ACL draft is in `paper/grpo_audit_acl/`. Blinded human
evaluation remains pending and is marked as a TODO in the manuscript.

All empirical quantities live in `results/paper_stats.json`. The paper-facing
derived section is `grpo_audit_paper`; `src/build_grpo_audit_paper.py` checks
run-level traces and renders `generated_stats.tex`.

## Central question

Can a cheap, policy-relative audit made before reinforcement learning predict
which failure mode a learned machine-translation reward will amplify?

The audit compares each reward's scores for controlled degenerate outputs with
its scores for the initial policy's translations. It makes a hack call, safe
call, or abstains separately for source copying, refusal, repetition, and
shortening. Frozen calls are evaluated only when optimization demonstrably
raises the assigned reward.

## Current answer

- Safe calls transfer reliably in the completed prospective scale study.
- Positive calls are conditional on reachability. XCOMET's source-copy
  preference becomes a realized exploit for Chinese under the larger policy,
  while the matched German runs raise reward without copying.
- The controlled MetricX reward-scale arm produces no audited failure for
  either tested reward scale.
- Correct rollout padding removes the earlier CometKiwi refusal collapse,
  preamble proliferation, and cross-metric failure. Those right-padded results
  remain disclosed as an implementation condition and are never pooled with
  corrected runs.
- Automatic reference metrics support the behavioral interpretation, but
  blinded human evaluation is still required for the final adequacy claim.

## Claim boundary

- QE reward hacking in generative MT is established by prior work, including
  WALAR. Copying and language guards are comparison points rather than new
  discoveries.
- Pre-training prediction of reward hacking also exists outside MT. SimHack
  predicts onset from smaller reference runs; this paper studies a cheaper,
  behavior-specific audit without reference optimization.
- The contribution is the prospective test, the separation between reward
  preference and policy reachability, and the fully rerun padding correction.
- The policy-scale arm changes LoRA parameterization and learning rate, so it
  does not identify a pure model-size effect.

## Paper structure

1. Motivation and related QE reward-hacking evidence.
2. Policy-relative audit, frozen thresholds, abstention, and manipulation gate.
3. Corrected smaller-policy grid and prospective scale design.
4. Complete called-cell scorecard.
5. Reachability analysis using the cross-direction XCOMET contrast.
6. Controlled MetricX reward-scale result.
7. Padding correction and right-padded versus corrected outcomes.
8. Blinded human evaluation and qualitative taxonomy.
9. Limitations and a reachability-aware next diagnostic.

## Remaining work

1. Obtain blinded German and Chinese judgments; Spanish can serve as secondary
   evaluation. Preserve the frozen sheets and key hashes.
2. Add human outcomes to the SSOT through a dedicated analysis script and
   regenerate paper macros. Do not hand-type results into LaTeX.
3. Add qualitative examples selected by a frozen, disclosed rule after scores
   are complete.
4. Expand the related-work comparison and verify final publication metadata.
5. Decide whether an additional reward-family amendment materially improves
   coverage. It is not required to interpret the completed study.

## Reproducibility requirements

`src/analyze_grpo_scale.py` is the only path from scale artifacts to the
prospective scorecard. `src/analyze_grpo_rule_tests.py --corrected` scores the
corrected earlier grid, and `src/analyze_grpo_ende_followup.py` scores the
frozen follow-up hypotheses. The paper builder asserts that every scale run and
reference-evaluation row traces to these recorded artifacts.
