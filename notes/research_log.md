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
