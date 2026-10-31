# Data and model notice

This project uses Tatoeba text and the Meta NLLB-200 distilled 600M model under
the terms identified below. The planned work is noncommercial academic
research. This notice summarizes the project's attribution and provenance; it
is not a legal opinion. The project researcher recorded a noncommercial-use
self-assessment in [`rights_clearance.json`](rights_clearance.json); this is not
institutional approval.

## Tatoeba English–German text

The candidate pairs were derived from Tatoeba's official detailed-sentence and
translation-link weekly exports dated 2026-10-31. Tatoeba identifies its text
sentences as CC-BY 2.0 FR by default. Some original sentences may separately be
CC0, but the detailed export used here has no per-sentence license column; the
project therefore records and follows the export-level CC-BY 2.0 FR terms for
all selected text. CC-BY 2.0 FR is an attribution license and does not itself
impose a noncommercial restriction. Audio is not used.

Tatoeba requires attribution to sentence authors. The local manifests preserve
each source/reference sentence ID and contributor name. If sentence text is
shared, retain those attributions. Project-level attribution:

> This project uses selected English–German text sentences from [Tatoeba](https://tatoeba.org), released under [CC-BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/). The data were filtered to direct English–German translation links whose two sentences each had at least one positive community review and no negative review; exact WMT23/WMT24 source/reference text matches were excluded. Text was NFC-normalized and stripped. Pair manifests were split by sentence ID and exact normalized text into train, calibration, development, and in-domain test partitions. Sentence IDs and contributor names are retained in the manifests.

The community-review filter is only a selection rule; it is not a guarantee of
translation quality. Detailed filters, source hashes, counts, and split hashes
are in `data/tatoeba_en_de/candidate_audit.json` and
`data/tatoeba_en_de/protocol_v1/split_audit.json`.

Official guidance: [Tatoeba downloads](https://tatoeba.org/en/downloads),
[using the corpus](https://en.wiki.tatoeba.org/articles/show/using-the-tatoeba-corpus),
and [attribution FAQ](https://en.wiki.tatoeba.org/articles/show/faq).

## NLLB model

- **Model:** Meta AI, `facebook/nllb-200-distilled-600M`.
- **Pinned repository revision:** `f8d333a098d19b4fd9a8b18f94170487ad3f821d`.
- **License:** [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- **Modification:** planned full-model policy-gradient fine-tuning for
  English→German on the Tatoeba training partition, comparing the untilted
  decoder policy with a fixed target-piece-length tilt. The training reward is
  sentence-level chrF++ as defined in [RL_PROTOCOL.md](RL_PROTOCOL.md).
- **Use and distribution:** the study is noncommercial research. The model
  license's noncommercial condition continues to apply to use and sharing of
  the model and modified weights. No model weights, optimizer states, or
  sentence-level data are included in this repository.

Give credit to Meta AI, identify the model and pinned revision, link the model
card and license, and describe the fine-tuning modification when reporting or
sharing permitted materials. The model card identifies NLLB-200 as a research
machine-translation model: [model card](https://huggingface.co/facebook/nllb-200-distilled-600M).

Run metadata records the model snapshot identifier and hashes for its config
and SentencePiece model. This notice describes the planned procedure; it does
not state that fine-tuning has occurred.
