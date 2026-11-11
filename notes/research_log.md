# Research log — reinforcement-learning

Earlier history (through 2026-11-03) is in the git log, `docs/*_PROTOCOL.md`,
and `results/*amendment*.json`.

## 2026-11-04

### /tmp wipe recovery
Login-node `/tmp` was wiped. Files that existed only there were copied to
`~/tmp_rescue_2026-11-04/reinforcement-learning/` (index: `../INDEX.md`).
- `wmt26_build_synthetic_references.py`: the only copy (upstream
  `wmt26-general-mt/synthetic_references/` has outputs but no script).
  Committed as `scripts/wmt26_build_synthetic_references.py`, with defaults
  re-pointed from the original wmt26-general-mt checkout layout to this repo.
  Its manifest (`scripts/wmt26_synthetic_manifest.json`) is identical to
  `data/reward_da/wmt26-filter-manifest.json`.
- `wmt25-genmt-humeval.jsonl`: byte-identical to the public LFS object in
  `wmt-conference/wmt25-general-mt` (sha256 a36ebce1…). Nothing reads it, so it
  was left in the rescue directory; provenance is in `data/reward_da/README.md`.
- `*_stdout.json`, `login01/*`, `login03/reward_paper_build*.json`: stdout of
  Oct 31–Nov 3 audits and builds. They stay in the rescue directory; the
  results they printed are already in `results/`.
- `grep "/tmp/"` over src/, scripts/, docs/, notes, results/ (incl.
  paper_stats.json): no input/output paths pointed at /tmp. Only hits are
  vLLM's own temp files inside today's failed judge logs.

### Main-track direction: LLM-judge validation
Status at start: Codex stopped (usage cap) while beginning a WMT24 en-de fresh
confirmation of the safe-pressure selector; nothing written. Decision: do not
run a third offline selector test. Lit check: fixed-pool "which metric is safe
to optimize" overlaps Perrella et al. EMNLP 2024 and Kovacs/Deutsch/Freitag
(arXiv 2411.03524). Open gap: does a cheap offline audit on human-rated pools
predict which real QE reward gets hacked under *generative* RL? That needs a
quality measure for new outputs, so step 1 is validating an LLM MQM judge.

- Protocol frozen before any judge output: `results/llm_judge_protocol.json`
  (sha 172360f1…). Only already-opened pools (WMT23 zh-en, WMT24 ja-zh,
  WMT24 en-es), 200 segments each, all systems (8,600 candidates). Judges:
  Qwen2.5-32B-Instruct, Gemma-2-27B-it, gpt-oss-20b. Selection by T1 only;
  acceptance A1–A4.
- Evaluator dry run (synthetic judges, real rewards + real human MQM): 0 of
  88 reward×pool×rule cells have significantly negative human selection gain,
  so A4 is untestable from WMT candidates (all are real MT outputs).
- Amendment 1 (`results/llm_judge_protocol_amendment_1.json`, sha 1792c5b1…),
  declared before reading any judge output: T4 known-hack detection over 5
  deterministic degenerations of the best human-rated candidate (source copy,
  half truncation, appended unrelated sentence, unrelated fluent output,
  repetition), 3,000 items; A5 = detection ≥ 0.90 per type.
- External QE scored: CometKiwi-22/23-XL on en-es, XCOMET-XL on all three
  pools, plus all three on T4 items.
- First judge array (15034272) failed: vLLM sampler warmup needs `ninja`, and
  the env bin was not on PATH. Fixed in the sbatch files; resubmitted as
  15034469 (validation) and 15034470 (T4).
- Resubmitted again (15034533 / 15034534) with `VLLM_USE_FLASHINFER_SAMPLER=0`:
  the dw A100 nodes have no `nvcc`, so FlashInfer's sm80 sampler JIT failed.
  All tasks completed.

### LLM-judge validation result: no judge accepted (frozen criteria)
- Parse rate: Qwen 0.997–1.0, Gemma 0.999–1.0, gpt-oss-20b 0.19–0.55
  (5,480 of 8,600 hit max_tokens=1024 while reasoning), so gpt-oss is
  ineligible.
- T1 within-source acc_eq: judges are roughly on par with the QE rewards they
  would referee (≈0.49–0.64; best per pool: CometKiwi-22 on zh-en, gpt-oss
  (parsed subset only) on en-es and ja-zh). T2 system-level Pearson is high
  (Qwen 0.95–0.98, Gemma 0.91–0.96).
- Selected (frozen rule) Gemma-2-27B: A1 pass, A2 pass (sign agreement 0.92
  over 77 significant cells), A3 FAIL (Spearman 0.30 < 0.60), A4 underpowered
  (0 significantly negative cells), A5 FAIL. Not accepted.
- T4 detection: Gemma misses source copies (0.02). Qwen detects copies (0.93)
  but misses truncation (0.71) and repetition (0.68). gpt-oss looks best on
  parsed items but is ineligible.
- Side finding (QE rewards on T4): XCOMET-XL in QE mode prefers the source copy
  over the best translation in 83.5% of segments (detection 0.165);
  CometKiwi-22 detects copies at only 0.71. CometKiwi-23-XL detects every
  type at ≥ 0.95.
- Interpretation: a ~30B local LLM judge is not a better segment-level referee
  than the rewards themselves, though it tracks system-level quality. Under the
  frozen protocol it cannot be used for headline claims.

### GRPO pilot launched; judge v2 also fails
- Prospective GRPO predictions frozen in `results/grpo_pilot_protocol.json`
  (commit b850cf9) before any GRPO code existed. P1: XCOMET-XL-QE run's
  untranslated rate rises by >= 0.10 and exceeds both CometKiwi runs. P2:
  CometKiwi-23-XL run's rate rises by <= 0.02. P3: ordering XCOMET >
  CK22 > CK23. Single seed, Qwen2.5-1.5B-Instruct en->es, 500 steps.
- GRPO env `~/.venvs/grpo` (torch 2.6 cu124, transformers 4.57.6, TRL 0.25.1,
  COMET). Needed `setuptools<81` (torchmetrics imports pkg_resources),
  `pyarrow==19.0.1` (newer pyarrow needs NumPy 2; COMET pins 1.26), and
  `OPENSSL_CONF=/dev/null` (uv's bundled OpenSSL can't parse the cluster's
  system config). Smoke test: ~34 s/step. Runs submitted as job 15035484.
- Judge v2 (`results/llm_judge_protocol_v2.json`): 450 segments disjoint from
  v1, gpt-oss-20b at 4096 tokens, hybrids with `src/hack_guards.py`, same
  thresholds. Result: not accepted. Selected Gemma-2-27B (judge-only, mean
  acc_eq 0.545; all six candidates within 0.535–0.545, vs 0.49–0.56 for
  the external QE rewards). A2 0.98 pass; A3 Spearman 0.599 (< 0.60) fail;
  A5 fail (source copy 0.018). gpt-oss still ineligible (parse 0.73–0.88).
  Only gpt-oss+guards would clear 0.90 on every T4 type. Guard
  false-positive rate on good real candidates: 0.001 zh-en, 0.010 en-es,
  0.060 ja-zh.
- Decision: stop iterating on judges (two frozen attempts; a third would be
  a forking path). Local ~30B judges are not better than the rewards at
  segment level. The GRPO predictions use deterministic indicators and do
  not need a judge. Quality claims need human evaluation.

## 2026-11-05

### GRPO pilot result: all three frozen predictions hold (single seed)
- Base Qwen2.5-1.5B-Instruct untranslated rate 0.028. XCOMET-XL-QE run:
  0.943 by step 100, 0.941 at step 500 (397/422 outputs). Manual spot check:
  the outputs are the English source copied verbatim. XCOMET scores them 0.92
  (base 0.69), while the held-out CometKiwi-23-XL falls 0.60 → 0.17.
- CometKiwi-22 run: untranslated 0.017; CometKiwi-23-XL run: 0.012. Both raise
  their own and held-out rewards modestly (e.g. the CK23 run moves XCOMET
  0.69 → 0.73 and CK22 0.745 → 0.765).
- P1, P2, P3 all true (`results/grpo_pilot_result.json`, SSOT `grpo_pilot`).
  Caveats: one seed; the CK22-vs-CK23 ordering in P3 rests on 0.017 vs 0.012;
  no human evaluation of the non-hacking runs.
- The offline T4 audit (XCOMET-XL prefers a source copy in 83.5% of segments)
  predicted the generative failure mode, its direction, and which rewards are
  safe. This is the paper's core result if it replicates.

### MetricX, en-de, and the home-quota failure (2026-11-05)
- Frozen audit→GRPO prediction rule (`results/grpo_audit_rule.json`): d_k < 0.50
  predicts a hack, d_k >= 0.95 predicts safe. Thresholds chosen after the pilot
  (disclosed); MetricX-24-QE and en-de are the real tests.
- `src/metricx_qe.py`: official-order MetricX-24 (EOS stripped per example
  before padding). AmanaMT's metricx.py strips after padding: a 2-item test
  gave 2.14 vs 0.85 for the shorter item. Recorded for the AmanaMT corrections.
- en-de protocol frozen (`results/grpo_ende_protocol.json`): 400 WMT24 en-de
  sources read outcome-blind; `<v>` span markup stripped and unused.
- All 9 replication/mitigation runs failed at the step-100 save: home (2 TB)
  was full. Failed partial dirs deleted. Pilot checkpoints verified-copied to
  `/nobackup/archive/usr/it238/reinforcement-learning/grpo_pilot/` and removed
  from home. All GRPO sbatches now write checkpoints under
  /nobackup/archive and copy eval files back to artifacts/. Home: 118 GB
  free afterwards; morphotok (712 GB) is the largest consumer.
- Resubmitted: MetricX audit on GPU (15066260; the CPU attempt was too slow
  in bf16), rule tests after it (15066261), replication + mitigation
  (15066262).

### Rule-test, replication and mitigation results (reviewed 2026-11-07; jobs finished 2026-11-05)
All 17 jobs COMPLETED. Rule frozen 14:37 UTC; MetricX audit finished 15:31 UTC.
- MetricX-24-QE audit d_k: repetition 0.418 (predict hack), source_copy 0.982
  and truncation 0.982 (predict safe).
- Replication: XCOMET copy hack in 3/3 seeds (untranslated 0.03 -> 0.89-0.94);
  CometKiwi-22 / 23-XL copy-safe in 3/3. Guard mitigation: guarded XCOMET
  stays at 0.009-0.012 in 3/3 seeds.
- Frozen-rule scorecard:
  - Hack predictions: XCOMET en-de copy CORRECT (0.028 -> 0.958). MetricX
    repetition WRONG in 4/4 runs (en-es seeds 0-2, en-de): rate stays <= 0.002.
    Reading: low d_k is not sufficient; the exploit has to be reachable
    from the initial policy (copying 2.8% at init, repetition 0.2%).
  - Safe predictions: all correct except CometKiwi-23-XL en-de truncation
    (d=0.973; 0.010 -> 0.292). This is a failure by the rule's letter.
- The CometKiwi-23-XL en-de mechanism is a REFUSAL collapse, not truncation:
  367/400 eval outputs at step 500 are the identical English string "Sorry,
  but I can't assist with that request. ..." (0 at steps 0-200, 342 at step
  300). Truncation flags only the 117 cases with long sources. Mean
  CometKiwi-23-XL on refusals is 0.462 vs 0.444 for step-0 translations;
  XCOMET 0.26, CometKiwi-22 0.34, MetricX 0.48 on the same refusals. Held-out
  own reward flat ~0.47 from step 200. No refusals in any other run.
  Implication: the audit's degeneration taxonomy (copy/truncation/
  repetition/unrelated) misses refusal / canned off-task English. This was
  found post hoc; any refusal audit added now is exploratory.
- Open: training-time reward curve for that run (logs on /nobackup);
  seeds 1-2 for en-de; numbers not yet in paper_stats.json.

### en-de seeds 1-2 and SSOT scorecard (2026-11-07)
- Refusal run training log (grpo_rule_15066261_5): train reward 0.33 -> 0.56
  over ~5.7 epochs; sharpest KL rise / entropy drop near epoch 3, matching the
  refusal takeover between eval steps 200 and 300. Held-out reward stays ~0.47,
  so the gain is on (repeated) training prompts.
- `src/analyze_grpo_rule_tests.py`: recomputes rates from eval_outputs.jsonl
  (asserted against summaries), scores the frozen rule, counts refusals
  (exploratory), merges `grpo_runs` + `grpo_rule_scorecard` into
  results/paper_stats.json. Current: hack 1/5, safe 11/12.
- First seed-1/2 submission (15131711) was rejected by the trainer's frozen-rule
  guard (en-de protocol froze seeds [0]). Added
  `results/grpo_ende_protocol_amendment_1.json` (seeds 1-2, all four rewards,
  disclosed as made after seeing seed-0 outcomes); trainer checks its hash.
  Resubmitted as 15131797 (array 7-14).

### Exploratory refusal audit (2026-11-07, job 15131901; post hoc)
`src/audit_refusal_exploratory.py` -> results/refusal_audit_exploratory.json
(+ SSOT key `refusal_audit_exploratory`). R0 = observed collapse string (fit to
outcome); R1-R3 generic. Source-copy repro matches frozen d (XCOMET 0.167 vs
0.165, CK22 0.722 vs 0.71, CK23 exact, MetricX 0.980 vs 0.982).
- A (audit-style d, refusal < best human): >= 0.93 for every reward x string,
  incl. CometKiwi-23-XL (R0 0.932). Adding a refusal type to T4 would have
  predicted SAFE for CK23 -> the audit would still have missed the collapse.
- B (refusal > step-0 policy output, eval sources): CK23 en-de R0 0.512
  (mean 0.47 vs step-0 0.44) — the only cell where a refusal beats the policy
  on average. CK23 en-es R0 0.209 (0.49 vs 0.60): consistent with no en-es
  collapse. Generic R1-R3 for CK23 en-de 0.025-0.185; all other rewards <= 0.09.
- Reading (post hoc, one case): the audit's comparator is wrong for this
  failure. T4 compares against the best human translation; GRPO compares
  against the policy's own samples. CK23 scores the initial en-de policy low
  (0.44), so a fluent off-task string becomes competitive. Candidate fix: a
  policy-relative audit statistic — must be frozen and tested prospectively
  (new pair / new reward) before any claim. Also R0's specific wording matters
  (generic refusals score much lower), so the exploit is string-specific.

### en-de seeds 1-2 results (jobs 15131797, finished 2026-11-07; reviewed 2026-11-09)
- Scorecard now complete (no missing runs): hack 3/9, safe 22/24.
  - XCOMET en-de copy hack in 3/3 seeds (untranslated 0.028 -> 0.927-0.958).
  - MetricX repetition hack predicted, never happens: 0/6 (en-es + en-de).
  - Safe misses: CK23 en-de truncation, seeds 0 and 1 (both refusal collapses).
- Refusal collapse recurs in CK23 en-de seed 1: 0 refusals through step 300,
  92 at 400, 353/400 at 500. DIFFERENT string from seed 0: "Sorry, but I
  cannot translate that sentence as it is incomplete and lacks context. Could
  you please provide more information..." (+ near-variants). Seed 2: no
  refusals, own reward 0.444 -> 0.492, normal German. Seed 1 started late
  (step 400), so seed 2 may simply not have reached it within 500 steps.
- CK22, MetricX: no refusals in any en-de seed.
- Takeaway: CK23 en-de refusal is reproducible (2/3) but the exact string
  varies by seed -> fixed degeneration lists can't enumerate it; supports a
  policy-relative or category-level (off-target-language) check.

### Follow-up queue + reference-based eval (2026-11-09)
- en-de amendment 2 (`results/grpo_ende_protocol_amendment_2.json`, sha 0062a877…):
  CometKiwi-23-XL, seeds 0-2 x {unguarded, GlotLID off-target guard}, 1000 steps,
  predictions H1-H3 frozen in the amendment. Job 15187608 (array 0-5, 16 h).
  Guard: reward - 1.0 if GlotLID top-1 != target (outputs >= 4 words). On existing
  outputs it flags 367/367 + 352/353 refusals; 10/400 en-de and 9/422 en-es step-0
  outputs (7 and 6 already untranslated). fasttext-wheel 0.9.2 + pybind11 added
  to ~/.venvs/grpo (numpy pinned 1.26.4; nothing else changed). Eval now also
  records off_target_rate (old indicators unchanged).
- Reference-based eval (`src/eval_grpo_reference.py`, job 15187719): chrF + COMET-22
  vs WMT24 refA from MTME v2 (not the MQM TSVs; en-de MQM stays sealed). 822/822
  eval segments align by globalSegId. Paired bootstrap CIs on final - step-0 COMET-22.
  - Hacks are real quality losses: XCOMET copy -0.10 to -0.13 COMET-22 (chrF
    -23 to -31); CK23 en-de refusal seeds 0/1 -0.33 (chrF -31 to -35).
  - Safe en-es runs improve modestly; every CI excludes 0: CK22 +0.012 to +0.015,
    CK23 +0.008 to +0.019, guarded XCOMET +0.013/+0.014, MetricX +0.020 to +0.024
    (MetricX chrF -0.9 to -1.7 though).
  - en-de: MetricX +0.023 to +0.026 (CIs exclude 0, chrF +1.1 to +1.4); CK22 and
    clean CK23 seed 2 flat on COMET-22 (CIs include 0) despite chrF +2.
  - The XCOMET guard turns a -0.105 hack into +0.013/+0.014 gains.
  - Re-eval of the follow-up runs is queued: job 15187870 (afterany:15187608).

### en-de amendment 2 results (job 15187608, finished 2026-11-09; reviewed 2026-11-10)
CometKiwi-23-XL, 1000 steps, 3 seeds x {unguarded, GlotLID guard}.
- H1 TRUE: unguarded refusal collapse 3/3 (step-1000 refusals 393, 371, 303/400),
  incl. seed 2 (clean in the 500-step protocol). Onset by step 200 in all
  three; the stretched LR schedule keeps the LR higher early, so this is
  not a pure horizon effect. Own reward 0.444 -> 0.587/0.527/0.475.
- H2 TRUE: guarded 0/400 refusals at every checkpoint, off-target 0.003,
  own reward 0.444 -> 0.509-0.513.
- H3 FALSE: guarded XCOMET (0.69 -> 0.73) and CK22 (0.665 -> 0.70) rise, but
  MetricX-QE ends below step 0 in all 3 seeds (0.812 -> 0.800/0.794/0.806).
- Reference eval (auto job 15187870): unguarded COMET-22 -0.27 to -0.36
  (chrF 46.7 -> 14-16); guarded -0.002 to +0.005, all CIs include 0
  (chrF +1.7 to +2.0). The guard prevents the loss but brings no clear COMET-22 gain.

### en-de amendment-2 follow-up results (jobs 15187608 + re-eval 15187870; reviewed 2026-11-09 evening)
Refusal counts (/400) at step 1000, unguarded: s0 393, s1 371, s2 303. Collapse begins by step 200
in all 3 (s0 378, s1 379, s2 199). Guarded: 0 refusals at every step, all seeds.
- H1 (>=200/400 in >=2/3 unguarded seeds): PASS, 3/3. Seed 2 (clean at 500 steps) collapses here, so
  the collapse is not seed-specific. Caveat: max_steps=1000 stretches the LR decay, so step 200 here
  runs at a higher LR than step 200 of the 500-step runs. This does not say "collapse is purely time".
- H2 (guarded off-target <= initial+0.02, CK23 >= 0.444): PASS, 3/3 (off-target 0.00, CK23 .509-.513).
- H3 (guarded XCOMET, CK22, MetricX each >= step 0, all seeds): FAIL. XCOMET .692 -> .731-.737 and
  CK22 .665 -> .700-.703 rise, but MetricX .812 -> .800/.794/.806 falls slightly in all 3.
- Reference eval (COMET-22 vs refA, paired bootstrap): unguarded -0.360/-0.332/-0.268 (chrF -31 to -33).
  Guarded +0.004/-0.002/+0.005, every CI includes 0 (chrF +1.6 to +2.0). The guard removes the
  collapse but gives no reference-measured gain.
- NEW (post hoc): guarded policies learn a German preamble, "Hier ist die Übersetzung ins Deutsche:".
  Counts (/400) at steps 0 / 500 / 1000: s0 2/156/181, s1 2/207/271, s2 2/173/231. It passes langid
  because it is German. It plausibly accounts for some of the CK23 gain and the length-ratio rise
  (1.19 -> 1.35-1.44); mean length +10%. Some outputs also add commentary ("Allerdings ist es wichtig
  zu beachten..."). Untested: rescore with the preamble stripped to see how much of the CK23 +.07
  it carries.

### en-zh prospective test of the policy-relative rule (job 15206048 + ref eval 15206062; reviewed 2026-11-10)
- Letter-of-rule tally: hack 3/3 correct, safe 30/33.
  - XCOMET copy hack predicted (p=0.895) and observed 3/3: untranslated 0.015 -> 0.74-0.82;
    COMET-22 -0.23 to -0.26 (CIs exclude 0).
  - CometKiwi-23-XL refusal predicted SAFE (p=0.025): 0 refusals in all 3 seeds. Consistent
    with the weak-initial-policy account (CK23 initial 0.633 on en-zh eval vs 0.444 en-de).
  - 3 "failures": XCOMET repetition (+0.11 to +0.14), all 3 seeds.
- Two indicator problems found post hoc (my error: en-de definitions frozen unadapted for Chinese):
  1. truncation_rate is saturated for Chinese: len(zh)/len(en) chars < 0.5 for 96% of step-0
     outputs. The 9 safe truncation cells pass trivially -> uninformative.
  2. Chinese repetition uses space-stripped char 6-grams; copied English trips it. 51/54, 48/49,
     53/58 XCOMET repetition flags are untranslated copies -> the copy hack, not a repetition hack.
  Corrected (post hoc) reading: hack 3/3, informative safe cells 21/21.
- Non-XCOMET runs improve COMET-22 in all 9 (+0.007 to +0.019; CK23 seed 2 lower bound +0.000)
  while chrF falls (-0.2 to -4.1; MetricX -3.6 to -4.1). Not yet understood; chrF on Chinese is
  character-level and may penalise rephrasing. Check whether MetricX outputs got shorter or longer.

### Preamble audit (exploratory, post hoc; job 15231013, 2026-11-10)
`src/audit_preamble_exploratory.py` -> results/preamble_audit_exploratory.json (+ SSOT key).
- Preambles at the final step (>= 10/400): every CometKiwi-22 run with >= 10, CK23 en-zh
  (90-130), CK23 en-de guarded (199-291), XCOMET en-de (56-119). MetricX: 0 in every run.
  Step-0 base rate: 9/400 en-de, 1/400 en-es and en-zh. Trailing-commentary pattern: 0 everywhere.
- Own reward with the preamble stripped:
  - CK23 en-de guarded: the only case where the preamble helps the reward. Gain +0.065..+0.069
    -> +0.059..+0.062 stripped, so it carries ~0.006-0.009 of the gain (~10-13%).
  - Every other run: stripping leaves the gain the same or RAISES it. The header is mildly
    penalised, i.e. a format habit, not an exploit.
- COMET-22 (refA) with the preamble stripped is >= raw in every run. Guarded en-de CK23:
  0.665-0.673 raw -> 0.689-0.695 stripped (step 0: 0.667). en-de CK22: 0.659-0.672 ->
  0.674-0.686. The reference eval had understated the quality of preamble-heavy runs.
  Caveat: step-0 outputs were not stripped (9/400 preambles in en-de); no CIs on stripped scores yet.
- en-zh chrF drop (CPU check): stripping preambles recovers at most +0.3. Outputs lengthen
  (69 -> 75-81 chars vs refs 71). CK23 keeps more Latin-script words (0.26 -> 0.47-0.55
  of outputs; refs 0.28). MetricX's -3.6..-4.1 chrF has no preamble or Latin-script
  explanation; it comes only with longer, rephrased output. Left as a chrF-vs-COMET-22
  disagreement for human evaluation to settle.
