# Rights and provenance review brief

**Status: pending institutional review.** No research-data model inference,
calibration, development check, or training should be run until the appropriate
BYU reviewer confirms the intended use below. Preparing this brief records the
questions and available provenance; it is not legal advice or clearance.

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
[facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M).
Its model card states CC-BY-NC-4.0. The project needs review of whether the
planned academic research, local HPC use, saved optimizer/model checkpoints,
and generated translations fit the applicable terms and BYU policies.

## OPUS-100

OPUS-100 is not part of this study's current data. Its
[Hugging Face card](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
lists the license as unknown, and the cached English-German rows lack upstream
corpus identifiers. It remains excluded unless a bounded provenance review
establishes usable terms for the relevant material.

## Questions for the institutional reviewer

1. Is the described use of Tatoeba under the recorded export-level CC-BY 2.0 FR
   terms acceptable, including local research training and evaluation?
2. Does the NLLB CC-BY-NC-4.0 condition allow the planned academic fine-tuning
   and evaluation on BYU HPC? Are additional institutional approvals needed?
3. What attribution should accompany internal reports, papers, or any later
   sharing of derived outputs?
4. May model checkpoints, optimizer states, and generated translations remain
   on the managed HPC under the proposed access controls? What limits apply to
   sharing them outside BYU?
5. Is a private, code-only Git repository acceptable under these terms, and
   what review is needed before making it public?

## Materials

- [RL protocol](RL_PROTOCOL.md), including split definitions and planned use.
- `data/tatoeba_en_de/candidate_audit.json` and
  `data/tatoeba_en_de/protocol_v1/split_audit.json` for compact local
  provenance, hashes, counts, and attribution-field documentation.
- The linked Tatoeba and NLLB terms pages above.

Record the reviewer's role, date, decision, scope, and any conditions here before
resuming model use. Do not record sensitive legal communications in this Git
repository.
