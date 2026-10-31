# Alternative intervention screen

> Historical screen of the earlier WMT23/OPUS-based framing. The active Tatoeba protocol and current gates are maintained in [RL_PROTOCOL.md](RL_PROTOCOL.md) and [GO_NO_GO.md](GO_NO_GO.md).

## Outcome

**No candidate is ready for an MT RL pilot.** The transformed policies can be sampled and scored consistently, but none of the tested settings gives a repeatable, useful target-token fertility shift while preserving initial translation behavior. These are decoder-policy procedure effects; none isolates fertility by itself.

## Ideas tested

| Idea | Calibration result | Assessment |
|---|---|---|
| Piece-length logit bias | Stronger settings (`β=.20/.30`) produced a clear token/word increase, with a 3.4 to 5.7 point drop in the custom character n-gram proxy and substantially shorter outputs. | Reject for the pilot: competence changes with the treatment. Details are in [GO_NO_GO.md](GO_NO_GO.md). |
| Mixture of baseline and length-biased policy | On the 128-row L40S screen, a 10% mixture had negligible fertility change (+.0046 tokens/word; 95% interval −.010 to +.019). A 25% mixture increased it by .046 (interval +.017 to +.079), with shorter outputs and a −1.58 proxy change (interval −3.43 to +.30). A same-device A100 check of 15% and 20% mixtures found near-zero fertility changes; their proxy changes were about −1.5 points. | The stronger mixture’s result was not stable across calibration runs; no useful competence-preserving dose established. |
| Continuation-piece bias | On the L40S screen, `β=.10` gave +.018 tokens/word (interval +.003 to +.037) and a small, uncertain proxy change (−.23; interval −1.49 to +1.02). Same-device A100 checks at `.15` and `.20` found no reliable fertility shift (intervals included zero). | The apparent `.10` shift needs independent replication; it did not extend to larger doses. Not ready for RL. |
| Baseline/biased-policy mixture on continuation pieces | A 25% mixture had effectively no fertility change (−.001 tokens/word; interval −.006 to +.003). | Competence was preserved, but there was no treatment separation. |
| SentencePiece BPE-dropout alternate paths | Exact alternate paths were available for 35/48 calibration references. The saved diagnostic reports 64.7 sampled tokens versus 47.6 canonical tokens on the covered examples (about 36% more). | Large token-count change, but the diagnostic samples a path after fixing the string and scores it afterward. It is not an on-policy RL intervention. |
| Swap to a cached tokenizer/model family | Cached mBART-50, M2M-100, NLLB-1.3B, and mT5 assets provide other tokenizers/models; a quick reference-side tokenizer screen showed different fertility. | A model swap also changes pretraining, parameters, and initial competence. It cannot identify a tokenization effect by itself. No cached ByT5 candidate was found. |

For the transformed decoder policies, sampling-versus-rescoring errors stayed around `2e-5`, so the behavior probabilities were internally consistent. The quality measure is a custom character n-gram proxy over orders 1–6, not official chrF++.

## Parallel execution

The first six-condition policy screen ran serially on one L40S and took 9m07s. A four-task A100 Slurm array ran the conditions concurrently in about 3m30s–3m55s per task. A matched A100 baseline then completed in 2m54s. The A100 baseline was added after noticing that cross-GPU sampling can produce different trajectories; the final dose comparisons use the matched A100 baseline. No jobs remain queued or running.

## Recommendation

Do not start RL training with these interventions. The useful next step is to formalize a same-string path intervention whose joint sampling distribution and likelihood are exact, then validate it synthetically. BPE-dropout rescoring alone does not satisfy that requirement. If a policy change is allowed, the current length-bias family can be reported as a total decoder-policy effect, but its quality cost and dose instability make it a poor first learning experiment.

The finite exact same-string check is complete; see [SAME_STRING_INVARIANCE.md](SAME_STRING_INVARIANCE.md). It shows that exact joint likelihood alone is insufficient: with a string-only reward, ordinary policy-gradient updates are invariant to a separately sampled segmentation. Token or position weighting changes the estimator and must be framed as such. The remaining gate is a written mechanism and estimand for any proposed intervention; MT RL stays paused.

That mechanism/estimand review is now in [INTERVENTION_FEASIBILITY.md](INTERVENTION_FEASIBILITY.md). Stand-alone estimator analysis is not the preferred contribution. A real-model RL comparison is conditionally worth pursuing as the total effect of the existing normalized autoregressive length-tilt procedure; it must include changed decoder states, output distribution, and starting quality in the treatment. This reframes the question away from fertility-only causality. No RL training should start until the training loop, corpus subset, reward, and evaluation protocol are fixed.

The revised protocol is in [RL_PROTOCOL.md](RL_PROTOCOL.md). It labels the length tilt as a decoder-policy intervention rather than direct alternate-path assignment and distinguishes common-decoder learning outcomes from assigned-policy outputs. The trainer and preflight implementation now exist; the remaining gates are listed in [GO_NO_GO.md](GO_NO_GO.md).

## Artifacts

- Serial policy screen: `artifacts/diagnostics/nllb600m_policy_variants.json`
- Serial paired analysis: `artifacts/diagnostics/nllb600m_policy_variants_analysis.json`
- Matched A100 dose outputs: `artifacts/diagnostics/nllb600m_policy_variant_matched_cs_0.json` through `_4.json`
- Matched A100 paired analysis: `artifacts/diagnostics/nllb600m_policy_variant_array_analysis.json`
- Parallel worker and array scripts: `nllb_policy_variants.py`, `run_nllb_policy_variant_array_cs.sbatch`
- Paired analysis code: `analyze_policy_variants.py`, `analyze_policy_variant_array.py`
