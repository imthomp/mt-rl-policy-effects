# Human evaluation: study-team notes (never send to annotators)

Annotators receive only their `sheet.csv` and `docs/HUMAN_EVAL_ANNOTATOR_GUIDE.md`.

- Sheets (redesigned 2026-11-18): `policy7b_en_de`, `policy7b_en_zh`, `rewardscale_en_de`, `rewardscale_en_zh` (primary) and `grid1p5b_en_es` (secondary). Send each annotator only `sheet.csv` plus `docs/HUMAN_EVAL_ANNOTATOR_GUIDE.md`.
- The system key is in `artifacts/human_eval/<sheet>/key.json`. Its sha256, along with that of `sheet.csv`, is recorded in `artifacts/human_eval/manifest.json` before any sheet is sent out.
- Annotators must not see the key, the hypotheses, or which conditions are involved.
- Ideally there are 2 annotators per pair, for agreement. If the project researcher annotates a pair, record that and treat that pair as secondary evidence.
- Planned analysis:
  - Per system, the mean score difference from the initial policy, with a paired item bootstrap.
  - Flag rates per system.
  - Agreement between COMET-22 and the human scores.
  - Scores with and without `extra_text`.
