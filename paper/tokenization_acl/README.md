# Tokenization ACL note

This short paper wraps up the project's tokenization branch as two bounded
results: exact same-string estimator invariance and a negative NLLB length-tilt
feasibility screen. It does not claim an RL training effect.

The paper-facing source of truth is `results/paper_stats.json` under
`tokenization_paper`. Run from the repository root:

```bash
python src/build_tokenization_paper_stats.py
python scripts/check_tex_numbers.py paper/tokenization_acl/main.tex
make -C paper/tokenization_acl
```

If `latexmk` is unavailable but Tectonic is installed, use
`make -C paper/tokenization_acl tectonic`.

`generated_stats.tex` is regenerated from the JSON and must not be edited.
The ACL style files are unmodified copies from the official
`acl-org/acl-style-files` repository at revision
`d5adc823ff0f80f98c80405ca0ab66c68e684409`. Their SHA-256 hashes are:

- `acl.sty`: `19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de`
- `acl_natbib.bst`: `6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e`
