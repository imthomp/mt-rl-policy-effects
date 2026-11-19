# Prospective GRPO reward-audit ACL draft

This manuscript reports the corrected generative GRPO studies: the scale study and the generalization study. It is a results draft, not yet submission-ready:
- blinded human evaluation is still an explicit TODO in `main.tex`;
- the abstract and outline are drafts for the author to revise (see `docs/GRPO_PAPER_OUTLINE.md`).

Every empirical quantity comes from `results/paper_stats.json` through `src/build_grpo_audit_paper.py`. Never edit `generated_stats.tex` or `generated_scale_runs.tex` by hand. The builder asserts that:
- every scale, generalization and reference-evaluation row traces to a recorded run;
- the figure is current;
- the text's assumptions still hold, for example that the reachability probe failed calibration.

Build from the repository root, in this order:

```bash
module load texlive/2024
python src/diagnose_truncation_flags.py
~/.venvs/grpo/bin/python src/plot_audit_outcomes.py
python src/build_grpo_audit_paper.py
python scripts/check_tex_numbers.py paper/grpo_audit_acl/main.tex
make -C paper/grpo_audit_acl
```

## Current claim

The audit is a reliable screen for ruling out named failures:
- safe calls held in every scale-study task;
- in the generalization study, every safe-call failure was the truncation indicator, mostly flagging cleanups of bloated initial outputs.

Hack calls came true in 7 of 8 tasks across the two studies. The one miss (XCOMET-XL, 7B, en-de) is not explained by how often the initial policy samples copies (the reachability probe failed calibration). It is explained by a small copy margin over translations the reward already rates highly. That copy-margin explanation is post hoc, and a fresh test is in progress (`results/margin_test_protocol.json`).

Correcting the rollout padding removed the earlier refusal and preamble failures. They are kept as a disclosed implementation condition.
