# GRPO reward-audit paper: outline (draft 2026-11-18, for Isaac to revise)

Target: ARR, Jan 4 2027 (ACL main track; Findings is the realistic fallback). Long paper, 8 pages plus appendix.

**One-sentence claim (candidate):** a cheap policy-relative audit, run before RL, reliably tells you which named failure modes a QE reward will *not* produce under GRPO, and predicts the ones it will produce only when the policy can reach the exploit.

**What is new vs prior work:**
- WALAR (2603.13045) shows QE hacking and fixes it after the fact. MiLMMT-46 (2608.10812) gates rewards with language ID. SimHack (2610.28707) forecasts hacking onset, but for code RL.
- Ours: *prospective, pre-training* prediction of the *failure type* for MT QE rewards. Predictions are frozen and scored at the task level across policies, rewards, reward-model scales and five language directions.

## 1. Introduction (~1 page)
- QE rewards make reference-free MT RL possible, and they get exploited (cite WALAR). Developers need to know *before* paying for a run.
- The audit in one paragraph; frozen calls; the headline scorecard (task level).
- Contributions:
  1. the audit and its frozen three-band rule;
  2. a prospective test across ~70 GRPO runs, with a manipulation check;
  3. the asymmetry: safe calls transfer, hack calls need reachability;
  4. the padding correction as a cautionary methods result.

## 2. Related work (~0.5 page)
- QE as reward and its failure modes: He et al. 2024, WALAR, MiLMMT-46, MT-R1-Zero.
- Predicting reward hacking (SimHack) and reward-model auditing.
- Metric robustness to degenerate outputs (the T4-style perturbation tests in metric meta-evaluation).

## 3. Policy-relative audit (~1 page)
- Definition of p_k; the four degeneration types and how each is built; why compare against the policy's own outputs rather than the best human translation (our first T4 rule failed for exactly this reason, so keep it as motivation).
- Three-band rule (hack > 0.50, safe <= 0.05, abstain otherwise); outcome margins; the manipulation check.
- Freezing procedure: hashes, the trainer gate, amendments.

## 4. Experimental setup (~1 page)
- Policies (Qwen2.5-1.5B full fine-tuning; 7B LoRA); GRPO settings table; English training prompts; 400 WMT24 eval sources shared across directions.
- Rewards: XCOMET-XL/XXL, CometKiwi-22, CometKiwi-23-XL/XXL, MetricX-24 large/XL/XXL.
- Directions: en-de, en-zh (scale study); en-ja, en-ru (generalization); en-es (corrected grid).
- Indicators (with prospective fixes) and reference metrics (COMET-22, chrF).

## 5. Results (~2.5 pages)
- 5.1 Scorecard: cell level AND task level, for the scale study, the generalization study and the corrected grid (table).
- 5.2 Safe calls transfer (all tasks), including refusal and the MetricX series.
- 5.3 Hack calls and reachability: XCOMET copying by direction and policy. Figure: audit p against the observed indicator change, one point per task, coloured by outcome. The en-ru call (p close to the failed en-de call) is the key prospective test.
- 5.4 Reward-model scale: MetricX large/XL/XXL and COMET XL/XXL.
- 5.5 Translation quality: COMET-22/chrF per condition; human evaluation (pending).

## 6. A spurious failure: rollout padding (~0.5 page)
- Right-padded rollouts manufactured a refusal collapse and preambles. Full rerun; what to log.

## 7. Discussion and limitations (~0.75 page)
- Named failures only; no reachability model (proposal: add a cheap reachability probe, e.g. the exploit's probability under the initial policy's sampling); correlated reward families; LoRA confound in the policy-scale arm; automatic metrics until human evaluation lands.

## Appendix
- Per-run tables (scale, generalization, corrected grid); audit p tables; the full right-padded vs corrected comparison; compute; prompts; human-evaluation protocol and guide.

## Framing decisions (made 2026-11-18, delegated by Isaac)
1. **Screen first; predictor only if earned.** The main claim is that the audit reliably rules out named failures. Hack prediction is claimed only for audit + reachability probe (results/reachability_probe_protocol.json), and only if its prospective test holds. If it fails, the paper is "a screen, plus a diagnosed limit". Committed before the probe's results.
2. **Keep the first audit (T4) briefly as motivation** (one paragraph + appendix: comparing against the human best failed, which motivates the policy-relative design). **Cut the bandit study** (a separate paper).
3. **Reachability goes in the method section** (frozen and tested prospectively), not just future work.

## Original open decisions (superseded by the section above)
- Framing: "a screen for ruling out failures" (what the evidence supports) vs "a predictor of hacking" (more ambitious, weaker evidence).
- Whether to keep the bandit/T4 history as motivation, or cut it entirely.
- Whether to propose the reachability probe as a method, which needs one more frozen test, or leave it as future work.
