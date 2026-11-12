# Reinforcement-learning class project proposal

`main.tex` is the historical course proposal. It follows the proposal fields in
the course directions, presents the completed finite-action contextual-bandit
study as preliminary evidence, and proposes a prospective test of a frozen
pre-optimization failure predictor. The repository now contains the subsequent
predictor experiments and generative GRPO extension; this directory preserves
the submitted proposal rather than serving as the current project overview.
Reported preliminary quantities are imported from the generated paper macros
backed by `results/paper_stats.json`.

Build with either:

```bash
make
```

or:

```bash
make tectonic
```

Run `make scan` before submission. The scanner flags numeric literals in the
hand-authored LaTeX source, helping keep experimental quantities tied to the
repository's results SSOT. The remaining literals in the preamble are
structural LaTeX settings covered by the shared scanner's exception rules.
