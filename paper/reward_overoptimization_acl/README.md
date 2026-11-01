# AmanaMT-derived reward-overoptimization ACL draft

This is an initial results manuscript, not a submission-ready paper. Its claim
is intentionally narrower than the original filtering hypothesis:

> Reward optimization is sensitive to reward-training composition, but the
> current evidence does not establish metric--annotation disagreement filtering
> as superior to matched data removal.

Paper quantities are derived into the `reward_overoptimization_paper` section
of `results/paper_stats.json` and rendered through `generated_stats.tex`.
Every table row has an asserted source trace in the JSON.

Build from the repository root:

```bash
python src/build_reward_overoptimization_paper.py
python src/plot_reward_overoptimization.py
python src/plot_contextual_bandit.py --output-stem paper/reward_overoptimization_acl/figures/contextual_bandit
python scripts/check_tex_numbers.py paper/reward_overoptimization_acl/main.tex
make -C paper/reward_overoptimization_acl
```

The package reuses the pinned official ACL style files in
`paper/tokenization_acl/`. If `latexmk` is unavailable, use the `tectonic`
Make target with a Tectonic binary.
