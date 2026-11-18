# Translation quality rating: annotator guidelines

Thank you for helping. Each item is one English source sentence or short paragraph, followed by 2–6 candidate translations into your language (Spanish, German or Chinese), labelled A, B, C and so on. The candidates are in random order, and you are not told where any of them came from.

## What to do for each candidate

1. **Score, 0–100 (`score_0_100`).** Rate how well the candidate conveys the meaning of the source as a fluent, natural translation.
   - **0** = no meaning preserved, or not a translation at all
   - **33** = some meaning preserved, but major errors or omissions
   - **66** = most meaning preserved; minor errors or awkwardness
   - **100** = perfect meaning and grammar

   Read every candidate for an item before scoring, so your scores within an item are consistent with each other. Ties are fine.
2. **Flags.** Put a `1` in the column if it applies; leave it blank otherwise.
   - `wrong_language`: mostly not in the target language
   - `untranslated`: mostly left in English, or copies the source
   - `refusal_or_not_translation`: refuses ("Sorry, I can't…"), asks a question, or answers something else instead of translating
   - `extra_text`: adds text that isn't a translation, such as "Here is the translation:", notes or explanations
3. **Comment (optional).** Anything notable.

**Extra text:** score the translation part on its own merits, then subtract what you think the added text costs a reader who needed just the translation. Always set the `extra_text` flag, so we can analyse it separately.

## Practicalities

- Each sheet has 60 items and roughly 220–290 candidates; expect about 2–2.5 hours per sheet. German and Chinese annotators get two sheets each. Breaks are fine.
- Do not look up the source online, and do not use MT tools.
- Return the completed `sheet.csv` without changing any other column.
- If something is unclear, write a comment and move on.
