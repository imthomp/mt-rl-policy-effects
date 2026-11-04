# Local DA diagnostic data

`diagnostic_sample.jsonl` is generated from the cached public
`RicardoRei/wmt-da-human-evaluation` Arrow shards by
`src/prepare_da_diagnostic_sample.py`. It contains dataset text and labels and
is intentionally excluded from Git. Its provenance, sampling rule, row counts,
and content hash live in `results/paper_stats.json`.

## WMT25 GenMT human evaluation (`wmt25-genmt-humeval.jsonl`, not cached here)

A copy recovered from login-node `/tmp` on 2026-11-04 is at
`~/tmp_rescue_2026-11-04/reinforcement-learning/wmt25-genmt-humeval.jsonl`
(128,674,273 bytes, sha256
`a36ebce10c9ea26f71449933721de3a87e27f6a5b09d96b5daa026eaf1d197ab`).  It is
byte-identical to the Git LFS object `data/wmt25-genmt-humeval.jsonl` in
`wmt-conference/wmt25-general-mt` (last changed in commit `e0cb47f`,
2025-10-07), so it can be re-downloaded.  No code in this repository reads it;
the WMT25 reward data used here is the MQM release in `wmt25-mqm.zip`.

## WMT26 synthetic references

`scripts/wmt26_build_synthetic_references.py` (recovered from `/tmp` on
2026-11-04) is the builder for the cESA document-selection/segment-repair
references described by `wmt26-filter-manifest.json`.  Its run manifest,
`scripts/wmt26_synthetic_manifest.json`, is identical to that file.
