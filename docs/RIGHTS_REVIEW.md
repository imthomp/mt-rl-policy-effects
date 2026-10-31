# Rights and provenance review brief

**Status: license gate cleared for the described noncommercial academic
research, based on the published terms summarized in
[DATA_MODEL_NOTICE.md](DATA_MODEL_NOTICE.md).** CC-BY 2.0 FR permits reuse of
Tatoeba text with attribution; CC-BY-NC-4.0 permits noncommercial use and
adaptation of the NLLB model subject to its terms. The licenses themselves do
not require university approval. This is a project reading of published terms,
not a formal legal determination; separate BYU policies or a commercial sponsor
may impose additional conditions. Keep the work noncommercial and follow the
attribution and model-license conditions below.

## Proposed work

- Academic feasibility study of English→German NLLB-200 distilled 600M policy-
  gradient fine-tuning using a small, sentence-disjoint Tatoeba sample.
- The frozen Tatoeba protocol has 2,048 training links, 1,024 calibration links,
  1,024 development links, and 2,048 in-domain test links. The six-run pilot is
  paused. OPUS-100 is excluded.
- Training, calibration, and evaluation would run on BYU's managed HPC. Model
  checkpoints and generated translations would be stored in the project's
  private HPC workspace. The proposed Git repository contains code and compact
  provenance metadata, not sentence text, model weights, or run outputs.
- No public distribution of dataset text, checkpoints, or generated outputs is
  planned in this protocol. Any later redistribution would need a separate
  review.

## Tatoeba data

The project uses the [Tatoeba weekly downloads](https://tatoeba.org/en/downloads)
and the [usage guidance](https://en.wiki.tatoeba.org/articles/show/using-the-tatoeba-corpus)
and [attribution FAQ](https://en.wiki.tatoeba.org/articles/show/faq). The weekly
text export is described as CC-BY 2.0 FR, with some sentences also available
under CC0. The downloaded detailed export has no per-sentence license field, so
the project conservatively records export-level CC-BY 2.0 FR and does not claim
that an individual row is CC0. Sentence IDs and contributor names are retained
in the local manifests; the split audit records hashes and counts, not sentence
text. The candidate audit records the export provenance and attribution fields.

The selected English→German links required positive community review and no
negative review on either sentence. This is a data-selection rule, not a
guarantee of translation accuracy or independent rights clearance.

## Model

The proposed local checkpoint is
[facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M),
pinned to revision `f8d333a098d19b4fd9a8b18f94170487ad3f821d`. Its model card
states CC-BY-NC-4.0 and identifies research as an intended use. The planned
work is noncommercial academic research. The model and any modified weights
remain subject to the license's noncommercial condition. The project will not
distribute model weights, optimizer states, or sentence-level data in the code
repository. Check for separate BYU policy before any commercially sponsored
project or external distribution.

## OPUS-100

OPUS-100 is not part of this study's current data. Its
[Hugging Face card](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
lists the license as unknown, and the cached English-German rows lack upstream
corpus identifiers. It remains excluded unless a bounded provenance review
establishes usable terms for the relevant material.

## Attribution and scope checks

1. Keep Tatoeba sentence IDs and contributor names in local manifests; include
   Tatoeba, its source link, CC-BY 2.0 FR link, and preprocessing changes in
   reports or any permitted data sharing.
2. Credit Meta AI and the pinned NLLB revision, link the model card and
   CC-BY-NC-4.0 license, and describe the fine-tuning modification when
   reporting or sharing permitted materials.
3. Keep model use and any sharing of modified weights noncommercial. Do not
   treat this notice as approval for a commercially sponsored project.

## Materials

- [Data/model notice](DATA_MODEL_NOTICE.md), including attribution text,
  preprocessing, the pinned model revision, and the fine-tuning description.
- [RL protocol](RL_PROTOCOL.md), including split definitions and planned use.
- `data/tatoeba_en_de/candidate_audit.json` and
  `data/tatoeba_en_de/protocol_v1/split_audit.json` for compact local
  provenance, hashes, counts, and attribution-field documentation.
- The linked Tatoeba and NLLB terms pages above.

If a separate institutional policy review occurs, record its scope and
conditions here without recording sensitive legal communications in this Git
repository.
