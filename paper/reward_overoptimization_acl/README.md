# AmanaMT-derived reward-overoptimization ACL draft

This is an initial results manuscript, not a submission-ready paper. Its claim
is intentionally narrower than the original filtering hypothesis:

> Every tested learned reward loses human-rated quality under sufficient
> optimization. Reward seed changes when failure arrives, and the current
> evidence does not establish metric--annotation disagreement filtering as a
> remedy.

Paper quantities are derived into the `reward_overoptimization_paper` section
of `results/paper_stats.json` and rendered through `generated_stats.tex`.
Every table row has an asserted source trace in the JSON.

Build from the repository root:

```bash
python src/build_reward_overoptimization_paper.py
python src/plot_reward_overoptimization.py
python src/plot_contextual_bandit.py --output-stem paper/reward_overoptimization_acl/figures/contextual_bandit
python src/plot_text_policy_bandit.py --output-stem paper/reward_overoptimization_acl/figures/text_policy_bandit
python scripts/check_tex_numbers.py paper/reward_overoptimization_acl/main.tex
make -C paper/reward_overoptimization_acl
```

If raw long-horizon or reward-seed results change, first run
`src/analyze_bandit_horizon.py` and `src/analyze_bandit_seeds.py`. The latter
uses only NumPy. The source--candidate policy is independently reconstructed by
`src/analyze_text_policy_bandit.py`.

The package reuses the pinned official ACL style files in
`paper/tokenization_acl/`. If `latexmk` is unavailable, use the `tectonic`
Make target with a Tectonic binary.
