# Generative GRPO paper integration plan

Status: outcome independent plan written after the corrected scale launch and
before any corrected scale run completed.

## Central question

Can a cheap, policy relative audit made before reinforcement learning predict
which failure mode a learned machine translation reward will amplify?

The audit compares each reward's scores for controlled degenerate outputs with
its scores for the initial policy's translations. It produces a hack call, a
safe call, or abstains separately for source copying, refusal, repetition, and
shortening. The prospective scale experiment tests those calls under GRPO.

## Claim boundary relative to prior work

- Do not claim that QE reward hacking in generative MT is new. WALAR reports
  that optimization can amplify holes in QE metrics and uses language and word
  alignment safeguards.
- Do not claim that pre-training prediction of reward hacking is new in
  general. SimHack predicts hacking onset in code RL using smaller reference
  runs and a learned dynamics simulator.
- MiLMMT-46 shows that reference-free GRPO can work at multilingual scale when
  the reward combines QE models and language identification, followed by
  checkpoint interpolation.
- The candidate contribution is a simple, interpretable, policy relative audit
  that predicts the *type* of MT reward exploit before the target run, with
  prospective tests across policies, rewards, language pairs, and reward-model
  scales. Whether this claim survives is determined by the frozen scorecard.

## Results that decide the paper story

1. **Prospective prediction:** report every frozen call, abstention, failed
   manipulation check, and outcome. The unit is a reward by seed by failure
   type cell; repeated seeds do not create independent audit probabilities.
2. **Policy scale:** compare the initial-policy audit and outcomes for the
   larger policy with the earlier smaller-policy runs. Avoid attributing a
   difference to model size when optimization settings also differ.
3. **Reward scale:** compare MetricX XL and XXL under the same small policy,
   language, seeds, and training settings. This is the clean scale contrast.
4. **Known hard case:** CometKiwi-23 on English to German is a prospective safe
   call for refusal despite an earlier smaller-policy refusal collapse. A new
   collapse is a high-value falsification, not a result to hide.
5. **Manipulation:** safe outcomes count only when the assigned reward rises by
   the frozen minimum. Otherwise the run is uninformative about resistance to
   optimization.
6. **Translation quality:** distinguish exploit detection from translation
   quality. Use the frozen automatic comparisons and blinded human evaluation;
   do not infer quality from the optimized QE reward.

## Paper structure after outcomes

1. Motivation and prior evidence from the exact finite-action test bed.
2. Generative MT setup and the policy relative audit.
3. Frozen prediction rule, abstention region, and manipulation check.
4. Prospective GRPO scorecard.
5. Policy and reward scale analyses.
6. Output guards as mitigation, with common-decoder and preamble-stripped
   quality comparisons where available.
7. Human evaluation and qualitative failure taxonomy.
8. Limitations: high-resource directions, two policy sizes, correlated
   reward families, finite training horizon, repeated-seed dependence, and
   audit templates that cover only named failure modes.

The existing fixed-candidate bandit work should become motivation and a compact
controlled study rather than compete with the generative experiment for the
main narrative. The tokenization study remains a separate negative-result
paper.

## Outcome-contingent decisions

- If hack calls replicate and safe calls mostly hold, center the paper on
  prospective exploit prediction and abstention.
- If CometKiwi refusal or another safe cell fails, center the paper on where a
  cheap audit transfers and where it does not; analyze why the initial audit
  missed the policy-generated exploit.
- If manipulation checks fail broadly, the scale study is underpowered as an
  audit test. Report execution but do not score apparent safety as success.
- If all calls fail despite manipulation, the audit is a prospective negative
  result. Retain it only if the failure analysis yields a concrete replacement
  diagnostic.
- If MetricX XXL behaves differently from XL, emphasize the controlled reward
  scale result. If it does not, discuss the policy and data dependence of the
  WALAR comparison rather than claiming a contradiction.

## Required artifact path

`src/analyze_grpo_scale.py` must be the only path from run artifacts to the
scale scorecard. It verifies frozen hashes, amendment metadata, one-to-one
source records, row-derived rates, and the manipulation gate, then merges the
complete result into `results/paper_stats.json`. Paper prose and tables must use
macros generated from that SSOT.
