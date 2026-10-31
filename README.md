# NLLB policy-gradient feasibility study

Research code and protocol notes for a proposed English-to-German NLLB-200
policy-gradient study. This repository is under active development. The six-run
training pilot is paused; see [GO/No-Go](docs/GO_NO_GO.md) and
[the protocol](docs/RL_PROTOCOL.md) for the current decision and open gates.

## Current design

The protocol uses sentence-disjoint Tatoeba splits and compares an untilted
rollout policy with a fixed token-piece-length tilt. The estimand is the total
decoder-policy-procedure effect under common decoding, not a fertility-only
effect. A temperature-only calibration diagnostic is included to assess whether
the proposed treatment asks a distinct question from ordinary temperature
changes.

## Repository contents

- `src/nllb_policy_rl.py`: policy-gradient trainer and preflight entry point.
- `src/evaluate_nllb_policy.py` and `src/summarize_nllb_policy_evaluations.py`: common
  and stochastic checkpoint evaluation with source grouping and multiple
  references.
- `src/calibrate_tatoeba_policy.py`: calibration-only entropy/temperature
  comparison; it is not a third training arm.
- `src/merge_tatoeba_calibration_replicates.py`: combines the sole prespecified
  extra rollout sample only when the initial quality interval is inconclusive.
- `src/prepare_tatoeba_manifests.py` and `src/prepare_tatoeba_protocol_splits.py`:
  candidate selection, attribution, and sentence-disjoint split builders.
- `tests/`: small synthetic estimator checks.
- `scripts/slurm/`: the active calibration and evaluation wrappers.
- `artifacts/`: local-only diagnostics, logs, and run outputs.
- `docs/`: protocol, go/no-go status, and research notes.

Downloaded corpora, sentence-level manifests, model checkpoints, run logs, and
diagnostic outputs are intentionally excluded by `.gitignore`. Compact
provenance and split-audit JSON files are retained; no model weights or
Tatoeba sentence text belong in this repository. The data builders expect
the official source exports and local WMT overlap files to be available on the
machine; set the corresponding `WMT24_XML`, `WMT23_SOURCE`, and
`WMT23_REFERENCE` environment variables when their default locations do not
apply. The OPUS-100 builder is retained for provenance work, but OPUS-100 is
currently excluded from the active protocol pending source-level clearance.

## Environment

The recorded BYU cluster environment uses PyTorch 2.11.0+cu130, Transformers
5.14.1, SentencePiece 0.2.2, and SacreBLEU 2.6.0. The scorer pins are in
`requirements-rl.txt`; the OPUS Parquet reader pin is in
`requirements-data.txt`. Model access is local-only in the Slurm jobs. Set
`NLLB_MODEL_DIR` to the approved local snapshot path.

The resource plan in the current protocol is one A100 job at partition/QoS
`cs/cs`, serial execution, and a 72 GiB PyTorch allocator cap. The published
license terms appear to support the described noncommercial study subject to
attribution; the project researcher recorded a noncommercial-use self-assessment in
`docs/rights_clearance.json`. This is not institutional approval. Model-work scripts require that project decision. Afterward, calibration, save/reload verification, the paired
development check, and final protocol freeze remain before the six-run pilot.
See [the rights note](docs/RIGHTS_REVIEW.md),
[the data/model notice](docs/DATA_MODEL_NOTICE.md), and
[the protocol](docs/RL_PROTOCOL.md).

The tracked Slurm wrappers cover evaluation smoke, calibration, checkpoint
round-trip verification, and the four-update paired development check.
They use the BYU `cs/cs` partition and accept `NLLB_MODEL_DIR` and
`NLLB_PYTHON` overrides. Other exploratory cluster wrappers are preserved in
`scripts/archive/` locally and excluded from the Git snapshot.

## Rights and reuse

No code license is assigned yet. Confirm repository ownership and choose a
license before making this repository public. The noncommercial research
license basis and required notices for the current data/model are summarized in
[DATA_MODEL_NOTICE.md](docs/DATA_MODEL_NOTICE.md); Tatoeba contributor
attribution must be preserved if text is redistributed. OPUS-100 remains
excluded.
