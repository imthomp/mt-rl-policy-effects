# HANDOFF: reinforcement-learning (RL MT reward hacking)

Single entry point for any agent or person resuming this project. **Rewrite the CURRENT STATE block whenever anything changes.** History lives in `notes/research_log.md` and `git log`; the plan for the paper is in `docs/GRPO_PAPER_PLAN.md`.

## CURRENT STATE (updated 2026-11-12, afternoon)

### Question
Can a cheap audit, run before RL, predict which failure mode a learned MT QE reward will amplify under GRPO? When it can't, do simple output guards catch what it misses? Copy and wrong-language hacks under GRPO are already published (WALAR, arXiv 2603.13045), so the candidate contribution is *prediction*. See `notes/lit_review_2026-11-11.md`.

### The blocking issue: right-padded rollouts
- **Every 1.5B GRPO training run before commit `7e0c59d` sampled rollouts from right-padded prompts.** TRL 0.25.1 drops `padding_side` inside `apply_chat_template`, and Qwen2.5 pads right by default. Codex found it; it is documented in `results/grpo_padding_rerun_amendment.json`.
- **Affected:** the pilot, replication and guard runs, MetricX en-es, the en-de rule tests, the en-de 1000-step follow-up, and en-zh. That is every headline 1.5B result in the research log up to 2026-11-11, including copying in 9/9 XCOMET runs, the CometKiwi-23 en-de refusal collapse, the language-guard result and the preambles.
  - These outputs (`artifacts/grpo_pilot`, `grpo_replication`, `grpo_rule_tests`, `grpo_ende_followup`) are now the **"right-padded rollouts" condition**. Keep them, disclose them, and never pool them with corrected runs.
- **Not affected:** evaluation and all audits (left-padded), the frozen rules and predictions, and the corrected scale runs.
- **Treat every pre-fix GRPO outcome as unconfirmed until its v2 rerun exists.** The refusal collapse is the most suspect: one seed's refusal says the sentence "is incomplete and lacks context".

### Running
| Job | What | Outputs |
|---|---|---|
| `15275870` (`RL_v2_rerun`, array 0-44) | 39/45 done at last check (the 6 running are the 1000-step follow-ups). Left-padded rerun of the full 1.5B grid: same protocols, seeds and frozen predictions | `artifacts/grpo_v2/<group>/<name>`; checkpoints in `/nobackup/archive/usr/it238/reinforcement-learning/grpo_v2/` |
| `15263954` (`RL_scale`) | **DONE and scored** (results/grpo_scale_result.json, SSOT `grpo_scale`): safe 84/84, hack 3/6 (7B en-zh XCOMET copies; 7B en-de XCOMET does not), all 28 pass the manipulation check | `artifacts/grpo_scale/` (eval files); checkpoints in `/nobackup/.../grpo_scale_v2/` |

### Next steps, in order
1. ~~Score the scale protocol~~ done 2026-11-12. Still to do: reference-based quality (COMET-22, chrF) of the scale runs; add `grpo_scale` to `GROUPS` in `src/eval_grpo_reference.py`.
2. **When `15275870` finishes,** rescore with the existing frozen scripts pointed at `artifacts/grpo_v2`:
   - `src/analyze_grpo_rule_tests.py` (T4 rule and en-zh policy-relative rule; paths need a v2 switch)
   - the amendment-2 H1-H3 checks
   - `src/eval_grpo_reference.py` and `src/eval_stripped_reference.py` (COMET-22 and chrF)
   - `src/audit_preamble_exploratory.py`

   Report right-padded and corrected results side by side. Do **not** change any rule or prediction.
3. **Optional amendment:** XCOMET-XXL and CometKiwi-23-XXL as new rewards. Both are downloaded to `/nobackup/archive/usr/it238/hf_cache`, and the user has CometKiwi-XXL access. XCOMET-XXL needs an **H200** (out of memory on an A100-80GB beside training; ~52 s/step on H200). Freeze the audit and predictions before any run.
4. **Human evaluation:** sheets are built and blinded (`src/prepare_human_eval.py`, `docs/HUMAN_EVAL_GUIDELINES.md`, key hashes in `artifacts/human_eval/manifest.json`). They currently draw on **right-padded** runs; regenerate from `grpo_v2` before sending them out. Annotators are needed for German and Chinese; the user could do Spanish as secondary evidence.
5. **Paper:** the user writes the abstract and outline first. Every number goes through `results/paper_stats.json` and generated macros (see `~/.claude/CLAUDE.md`). Write "CometKiwi-23", never "CK23".

### Conventions
- Slurm job names are `RL_<desc>`. Use the `dw,dwmatrix` / `matrix` queue first, but compare `sbatch --test-only` across `cs2/cs`, `cs/cs` and `m13h` before big launches.
- Every new experiment is frozen (protocol or amendment, hash pinned in `src/train_grpo_pilot.py`) **before** its run. The trainer refuses unlisted runs.
- Checkpoints go to `/nobackup` (the 2 TB home quota filled once), with only eval files copied back to `artifacts/`.
- Environment: `~/.venvs/grpo` (torch 2.6, TRL 0.25.1, COMET, peft 0.17.1, fasttext-wheel). Set `OPENSSL_CONF=/dev/null` for HF downloads.
- Commits: never add Co-Authored-By lines. The remote is `imthomp/mt-rl-policy-effects` and pushes need the user's approval. Experimental artifacts are untracked.
