# Literature check, 2026-11-11 (positioning for ACL main track)

Read via arXiv abstract and HTML pages. Items marked [summary] came from a truncated full-text extraction and need verification against the PDF before citing.

## Closest prior work

**WALAR**: Liu, Ouyang, Hosmane Revanasiddappa, Li (CMU). "Mending the Holes: Mitigating Reward Hacking in Reinforcement Learning for Multilingual Translation." arXiv 2603.13045, Mar 2026. Code: LeiLiLab/WALAR.
- GRPO with a QE reward (MetricX-24-Hybrid-XXL, plus xCOMET per [summary]). Policies: LLaMAX3-8B, Qwen3-8B, TranslateGemma-4B; preliminary runs on Qwen2.5-0.5B.
- Reported hacks: the QE reward gives a perfect score to repeating the source (100% in preliminary QE-only training [summary]); wrong-language output (92% [summary]); over- and under-translation.
- Fix: a hybrid reward, i.e. QE + word-alignment F1 (BGE-M3), with a GlotLID+MaskLID language gate that sets the reward to -25 on wrong language.
- Evaluation is mostly the same metric family as the reward, plus spBLEU, a Gemini judge and a small human evaluation [summary].
- **No prediction or audit before training.** Hacks are diagnosed after they happen. Refusals and preambles are not mentioned.

**MiLMMT-46**: Han, Gao, Fu, Luan (Xiaomi). "Reference-Free Post-Training of Open LLMs for Multilingual MT." arXiv 2608.10812, Aug 2026.
- GRPO with reward = mean(XCOMET-XXL, CometKiwi-23-XXL) behind an OpenLID-v3 language gate (reward 0 on mismatch). Policies 1B–12B, 46 languages.
- Cites WALAR for reward hacking but includes no hacking analysis of its own. spBLEU falls 1.21 while the QE metrics rise.

**SimHack**: Deng, Bie, Ponomareva (Google Research). "Predicting the onset of detectable reward hacking before RL training." arXiv 2610.28707, Oct 2026.
- Code RL (LeetCode), not MT. It fits a behavioural simulator to small reference runs and predicts when hacking starts (t50) for new recipes.
- "The initial hacking rate alone does not determine onset" — this matches our MetricX-repetition and refusal findings.

**Other leads (not yet read)**
- A difficulty-adaptive MT RL paper (arXiv 2607.29287) reports that CometKiwi/COMET-only rewards cause BLEU to collapse ("copying source tokens") and adds a BLEU term.
- MT-R1-Zero (EMNLP 2025 Findings) uses GRPO with BLEU/CometKiwi rewards.
- Earlier MT work: Perrella et al. (EMNLP 2024); Kovacs, Deutsch and Freitag (arXiv 2411.03524).

## What this means for us

1. **Not novel:** that QE rewards get hacked by source copying or wrong-language output under GRPO (WALAR), and that a language-ID gate fixes wrong-language output (WALAR, MiLMMT). The copy hack and the language guard must be framed as **replication and tested mitigation**, not as discoveries.
2. **Novel, as far as these three papers show:**
   - **Predicting before training** which QE reward will be hacked, and how, from an offline audit, with predictions frozen and scored. SimHack is the only predictive work, and it covers code, not QE rewards.
   - **Reward-specific contrasts under one setup:** XCOMET copies; CometKiwi-23 refuses; MetricX-24-large showed no hack in any of 9 runs. That last result contrasts with WALAR, which reports source repetition with MetricX-24-XXL. So **scale of the reward model matters**, and adding MetricX-24-XXL as a reward tests this directly.
   - **The refusal collapse** (polite English refusals rewarded by CometKiwi-23) and its dependence on the strength of the initial policy. Not reported in any of the three.
   - **The preamble habit** and its effect on reference-based evaluation (stripping changes the conclusions).
3. **Reviewer risks:**
   - WALAR is a direct comparison point and must be cited prominently. Check whether it is published (ACL 2026?).
   - We use one policy family and small models. The planned 7B runs address this.
4. **To do:** read WALAR in full (PDF) and verify the [summary] numbers; read 2607.29287; search for "refusal" + QE reward and LLM-judge reward hacking in MT.


## Verification against the source (2026-11-18)
Checked against the WALAR PDF text (pypdf extraction) and the ACL Anthology; this section supersedes the [summary] items above.
- **Venue:** WALAR is **Findings of ACL 2026**, pp. 33699–33723, `2026.findings-acl.1682`. On the Anthology the third author is "Yatish H R".
- **Setup:** reward MetricX-24-Hybrid-XXL-bf16. GRPO in OpenRLHF: 8 rollouts, temperature 1, ε 0.2, KL β 0.01, peak LR 5e-7.
- **CORRECTION:** the "100%" repetition is a **self-generated reference**: the policy repeats *its own hypothesis*, which flips the hybrid QE model into reference mode. It occurred **only in preliminary QE-only training of Qwen2.5-0.5B-Instruct, "not ... on larger base models like Qwen3-8B and LLaMAX3-8B-Alpaca"**. The intro does also say "repeating input source sentences". Our null MetricX-XL/XXL result at 1.5B is therefore not in tension with WALAR.
- **Wrong-language ratio:** 92.43% for LLaMAX + QE-only. Token length 40.00 vs the reference's 62.04.
- **Human evaluation:** WALAR preferred in 42% (Az-Pt) and 51% (En-Kn), comparable in 34% and 39%.
- **Not mentioned:** refusals, preambles, prediction before training.
- **Other bib entries verified:** SimHack 2610.28707, MiLMMT-46 2608.10812 (v2), DeepSeekMath 2402.03300, and He et al. 2024 (NAACL long, 2024.naacl-long.451, pp. 8164–8180).
