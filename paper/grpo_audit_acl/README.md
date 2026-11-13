# Prospective GRPO reward-audit ACL draft

This manuscript reports the corrected generative GRPO study. It is a current
results draft, not a submission-ready paper: blinded human evaluation remains
an explicit TODO in `main.tex`.

Every empirical quantity is derived from `results/paper_stats.json` by
`src/build_grpo_audit_paper.py`. The builder also asserts that every scale run
and reference-evaluation table row traces to a recorded source run. Do not edit
`generated_stats.tex` by hand.

Build from the repository root:

```bash
python src/build_grpo_audit_paper.py
python scripts/check_tex_numbers.py paper/grpo_audit_acl/main.tex
make -C paper/grpo_audit_acl
```

The current claim is bounded: the prospective audit made reliable safe calls,
but positive exploit calls transferred only when the trained policy reached the
audited behavior. Correcting rollout padding removed the earlier refusal and
preamble failures, which are retained as a disclosed implementation condition.
