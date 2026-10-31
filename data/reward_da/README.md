# Local DA diagnostic data

`diagnostic_sample.jsonl` is generated from the cached public
`RicardoRei/wmt-da-human-evaluation` Arrow shards by
`src/prepare_da_diagnostic_sample.py`. It contains dataset text and labels and
is intentionally excluded from Git. Its provenance, sampling rule, row counts,
and content hash live in `results/paper_stats.json`.
