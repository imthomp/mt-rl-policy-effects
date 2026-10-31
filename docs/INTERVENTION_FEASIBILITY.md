# Intervention feasibility decision

> Historical feasibility assessment from the OPUS-100 phase. It predates the sentence-disjoint Tatoeba protocol and the later A100 memory checks. Use [GO_NO_GO.md](GO_NO_GO.md) for current status.

## Decision

**The estimator-focused study is complete as groundwork, but is not strong enough by itself as the intended MT RL contribution. A bounded real-model study is worth pursuing only as the total effect of a segmentation-oriented autoregressive policy procedure.** Do not describe its result as the causal effect of fertility alone or as an isolated representation effect.

**No research-data RL training is ready to start yet.** The decoder-policy intervention and protocol are specified, and the rollout/loss entry point exists. The OPUS-100 manifests and exact overlap audit are complete; a one-batch actual-data preflight passed on an A100 80 GB. That preflight peaked at 55.43 GiB, so the original L40S 48 GB plan does not fit. A 256-action worst-case memory check remains, and licensing for the exact OPUS-100 source data has not been verified. The Hugging Face dataset card currently marks the license unknown, while OPUS-100 combines many source corpora rather than supplying one clearly established corpus-wide license ([dataset card](https://huggingface.co/datasets/Helsinki-NLP/opus-100), [OPUS-100 description](https://opus.nlpl.eu/opus-100.php)).

## What changes inside NLLB

The available checkpoint is NLLB-200 distilled 600M, English→German. The candidate intervention already exercised in this project is the per-token transform

```text
πβ(token | prefix, source)
  = softmax((logits + β · (1 − visible_piece_codepoint_length)) / temperature)
```

with special tokens unshifted and the softmax normalization recomputed at every decoder step. Sampling and rescoring use this same normalized policy. At `β=0.20`, the policy increases the relative selection of shorter visible SentencePiece pieces. Each selected token becomes the next decoder input. Consequently, token identity and count change the decoder's embeddings, positions, causal self-attention/KV states, future token distributions, EOS timing, and action horizon. The resulting surface-string distribution changes too. This is a **total decoder-policy intervention**, not a procedure that only changes a segmentation label for a fixed model state.

An alternate segmentation path is therefore not representation-neutral inside this model. Even when two paths decode to the same text, they feed different token IDs and lengths through the autoregressive decoder. The same-vocabulary property controls the available embedding/output rows; it does not establish that the model saw each alternate sequence or learned equivalent states.

## Familiarity and baseline competence

| Evidence | What it supports | What it does not establish |
|---|---|---|
| NLLB's released tokenizer uses a shared SentencePiece BPE model; the vocabulary is available with the 600M checkpoint. | Candidate paths can reuse model token IDs and parameters; no new embedding/output rows are required. | That alternate paths were common in NLLB pretraining or have familiar contextual representations. NLLB training used its chosen tokenizer segmentation; the project has no pretraining path-frequency record. |
| Existing fixed-reference diagnostic used SentencePiece BPE-dropout on 48 WMT23 references; exact round-trip coverage was 35/48. On covered references, sampled paths averaged 64.7 tokens vs 47.6 canonical (1.36×). | The tokenizer can produce valid alternate paths for a substantial but incomplete subset of this calibration sample. | On-policy generation, equal model competence, or broad corpus coverage. These paths were scored after fixing the reference string. |
| The same fixed-string diagnostic found mean path log probability per token of −8.56 for sampled paths and −9.26 for canonical paths. | The decoder assigns different token-level likelihoods to the two representations. | Familiarity in a behavioral sense or a fair sequence-probability comparison: token counts differ and the two autoregressive state trajectories differ. |
| Matched 128-example WMT23 calibration for the actual autoregressive length tilt found `β=.20`: +0.055 tokens/word (95% paired interval +0.022 to +0.088), custom character n-gram proxy −3.40 points (interval −5.76 to −1.23), about 32 fewer characters and 5 fewer words. At `β=.30`, the fertility shift rose to +0.069 while proxy quality fell by 5.66 points. | There is a measurable policy/fertility change and an initial competence/output-distribution cost. This cost is observable and can be included in a total-effect estimand. | A learning effect, final translation quality, or official chrF++ result. The proxy is custom and calibration-only. |
| Smaller/mixture alternatives preserved more baseline behavior but produced no stable, useful token-fertility shift in the matched screens. | Current attempts have not found a mild dose with both treatment separation and small calibration effects. | That no other intervention or dose can work. |

The relevant literature already establishes that sampling multiple segmentations with the same subword vocabulary can change NMT training and robustness. Kudo's subword regularization trains on probabilistically sampled segmentations; Provilkov et al.'s BPE-dropout uses a fixed BPE vocabulary and randomly drops merges, reporting supervised NMT gains and changes in learned token embeddings. That makes a generic “alternative segmentations help MT” claim insufficiently new. A distinct contribution here would need to be about **sequence-level RL learning under a changed autoregressive policy**, with the policy and outcome explicitly named. ([Kudo 2018](https://aclanthology.org/P18-1007/); [Provilkov et al. 2019](https://arxiv.org/abs/1910.13267); [NLLB-200](https://arxiv.org/abs/2207.04672).)

## Defensible question and estimand

> For NLLB-200 distilled 600M on English→German, what is the total effect on held-out reward learning of applying a fixed, normalized target-side length-tilted autoregressive policy during RL fine-tuning, compared with the native policy?

The treatment includes changes in token-path frequencies, decoder representations, output-string probabilities, length/EOS behavior, and starting translation quality under the transformed policy. It does **not** estimate a fertility-only effect. The token-per-word change is a treatment diagnostic, not the treatment itself.

A minimum comparison would clone the same checkpoint and optimizer initialization into native-policy and transformed-policy arms, use the same source examples, reward code, update budget, and paired seeds, and keep sampling and likelihood scoring under each arm's exact policy. Report both (a) each arm's own pre-to-post learning change under its own decoding policy and (b) a common canonical-decoder evaluation of both resulting parameter sets. Also report unadjusted updates and a separately calibrated matched-update-scale view; neither should be presented as controlling every optimization difference. Measure held-out string reward, an independent translation metric, validity/language errors, output length, EOS/truncation, fertility, policy movement, and variance across seeds. Do not tune on the reserved WMT24 set.

This comparison asks a useful operational question: does a fertility-oriented decoder policy help or hinder reward-driven learning when deployed as a whole? A negative result would reject that procedure for this task, not the broader hypothesis that tokenization can affect RL optimization.

## Feasibility gates before the pilot

1. Resolve licensing/provenance for the exact training pairs or choose a demonstrably licensed alternative; the current OPUS-100 card gives no dataset-wide license.
2. Review the policy-gradient loop, which differentiates the exact transformed log probability used for sampling. Synthetic and actual-data one-batch checks pass the `1e-4` behavior-score gate; the worst-case action-cap memory remains unverified.
3. Freeze the policy transform, reward, primary outcome, meaningful effect range, seed count, and update/compute budget before looking at held-out outcomes. Treat the initial quality difference as part of the total effect; do not reject the treatment merely because it fails exact competence matching.
4. The final frozen manifests and actual-data preflight are complete. The A100 80 GB preflight took 16.99 seconds for its update batch, but responses averaged 24 actions; stress memory up to the 256-action cap and select the final GPU/QoS before costing the six-run pilot.

Cached options include NLLB distilled 600M (4.7 GB), NLLB distilled 1.3B (11 GB), NLLB 3.3B (17 GB), M2M-100 1.2B (9.3 GB), mBART-50 (4.6 GB), and mT5 small/large (2.3/14 GB). No ByT5 checkpoint appeared in the cache inventory. NLLB-600M is the right starting point because the existing calibration and policy code already target it; switching model families would add pretraining, vocabulary, and competence differences without helping identify this intervention. The minimal RL loop now exists; data-use clearance and conservative long-sequence GPU memory remain unresolved.

## Stop/go rule

- **Go** to a small, explicitly exploratory two-policy RL pilot if the data and training-pipeline gates are resolved and the research claim is accepted as a total policy-procedure effect.
- **Stop this framing** if the paper must claim fertility alone caused the learning difference, or if the desired contribution is only a correctness proof of the response-mean invariance already documented in [SAME_STRING_INVARIANCE.md](SAME_STRING_INVARIANCE.md).
- **Current status: conditional go for the total-effect question; no-go for launching training now.** The existing dose/competence trade-off is a real treatment property, not a reason to demand impossible exact preservation. The decision remains limited to this model, language direction, and policy transform.

This is the historical feasibility assessment; its original remaining gates were superseded by the later Tatoeba protocol and preflight work. The current rollout–loss–reward–evaluation protocol is in [RL_PROTOCOL.md](RL_PROTOCOL.md). It specifies the estimands, common and assigned decoding evaluations, the fixed chrF++ reward and BLEU quality metric, and the six-run feasibility-pilot budget. Current gates are listed in [GO_NO_GO.md](GO_NO_GO.md). Training remains paused.
