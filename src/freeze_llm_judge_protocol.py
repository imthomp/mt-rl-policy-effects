"""Freeze the LLM-judge validation protocol before any judge output exists.

The judge is intended as a stand-in for human MQM when scoring generative-RL
outputs.  It is validated only on pools whose human MQM outcomes were already
opened by earlier prospective studies, so no fresh pool is consumed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from score_mqm_confirmation_candidates import load_candidates

ROOT = Path(__file__).resolve().parents[1]
MQM_ROOT = Path.home() / "projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation"
OUTPUT = ROOT / "results/llm_judge_protocol.json"
SAMPLE_SEED = 20261104
SEGMENTS_PER_POOL = 200

POOLS = {
    "wmt23_zh_en": {
        "source": "generalMT2023/zhen/mqm_generalMT2023_zhen.tsv",
        "manifest": "results/predictor_confirmation_manifest.json",
        "source_lang": "Chinese",
        "target_lang": "English",
    },
    "wmt24_ja_zh": {
        "source": "generalMT2024/mqm_generalMT2024_jazh.tsv",
        "manifest": "results/predictor_confirmation_manifest.json",
        "source_lang": "Japanese",
        "target_lang": "Chinese",
    },
    "wmt24_en_es_mqm": {
        "source": "generalMT2024/mqm_generalMT2024_enes.tsv",
        "manifest": "results/safe_pressure_manifest.json",
        "source_lang": "English",
        "target_lang": "Spanish",
    },
}

JUDGES = {
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "gemma-2-27b": "google/gemma-2-27b-it",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}

PROMPT_TEMPLATE = """You are an expert professional translator and translation quality annotator using the MQM (Multidimensional Quality Metrics) framework.

Annotate every error in the {target_lang} translation of the {source_lang} source below.

For each error give:
- "span": the erroneous text copied from the translation (or "" for an omission)
- "category": one of "accuracy/mistranslation", "accuracy/omission", "accuracy/addition", "accuracy/untranslated", "fluency/grammar", "fluency/spelling", "fluency/punctuation", "fluency/register", "fluency/inconsistency", "style/awkward", "terminology/inappropriate", "non-translation", "other"
- "severity": "minor" (does not impede understanding) or "major" (changes or obscures meaning, or a serious fluency problem). Use category "non-translation" only when the output is not a translation of the source at all.

{source_lang} source:
{source}

{target_lang} translation:
{hypothesis}

Respond with only a JSON object of the form {{"errors": [{{"span": "...", "category": "...", "severity": "..."}}]}}. Use {{"errors": []}} if the translation has no errors."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Protocol already frozen: {OUTPUT}")
    rng = np.random.default_rng(SAMPLE_SEED)
    pools = {}
    for name, spec in POOLS.items():
        metadata, rows = load_candidates(
            MQM_ROOT / spec["source"], ROOT / spec["manifest"], name
        )
        segments = sorted({row["segment_key"] for row in rows})
        systems = sorted({row["system_id"] for row in rows})
        chosen = sorted(
            rng.choice(len(segments), size=SEGMENTS_PER_POOL, replace=False).tolist()
        )
        sampled = [segments[i] for i in chosen]
        pools[name] = {
            **spec,
            "source_sha256": metadata["source_sha256"],
            "candidate_content_sha256": metadata["candidate_content_sha256"],
            "segments_available": len(segments),
            "systems": systems,
            "sampled_segments": sampled,
            "sampled_candidates": len(sampled) * len(systems),
        }
    protocol = {
        "status": "frozen_before_any_judge_output",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Validate an LLM MQM judge as a proxy for human MQM when scoring "
            "outputs of reward-optimized MT policies."
        ),
        "pool_rationale": (
            "All three pools already had human MQM opened by completed "
            "prospective studies, so they are no longer eligible as fresh "
            "confirmation data. Unused pools (e.g. WMT24 en-de) stay sealed."
        ),
        "sample_seed": SAMPLE_SEED,
        "segments_per_pool": SEGMENTS_PER_POOL,
        "pools": pools,
        "judges": JUDGES,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
        "decoding": {"temperature": 0.0, "max_tokens": 1024, "max_model_len": 4096},
        "scoring": (
            "Judge errors are scored with mqm_human_scores.error_penalty, the same "
            "weights as human MQM (major 5, minor 1, minor punctuation 0.1, "
            "non-translation 25); quality = -penalty. Unparseable outputs are "
            "counted, retried never, and excluded from analysis; parse rate is reported."
        ),
        "human_quality": "mqm_human_scores.load_mqm_quality (mean over raters, higher is better)",
        "tests": {
            "T0_parse_rate": "fraction of candidates with a parseable JSON error list",
            "T1_within_source_pairwise_accuracy": (
                "Over all candidate pairs sharing a source segment, tie-calibrated "
                "pairwise accuracy (acc_eq, Deutsch et al. 2023) of judge quality "
                "against human quality. Reported for every judge and for external "
                "QE rewards on the same candidates as reference points."
            ),
            "T2_system_level_pearson": "Pearson of per-system mean judge vs human quality within each pool",
            "T3_selection_gain_agreement": (
                "For every reward r available on a pool (all trained reward "
                "instances plus external QE controls) and each selection rule "
                "(top-1, exact expected best-of-4), compute the gain over the "
                "uniform-candidate mean in human quality and in judge quality. "
                "Units: within-pool z-scores of each quality measure. Uncertainty "
                "by segment bootstrap (2000 resamples, seed 20261104)."
            ),
        },
        "judge_selection_rule": (
            "Select the judge with the highest mean T1 acc_eq across the three "
            "pools among judges with T0 >= 0.95. T3 is not used for selection."
        ),
        "acceptance_criteria": {
            "A1": "selected judge T0 parse rate >= 0.95 on every pool",
            "A2": (
                "T3 sign agreement >= 0.80 over reward x pool x rule cells whose "
                "human-gain 95% bootstrap interval excludes zero"
            ),
            "A3": "T3 Spearman correlation between judge gain and human gain across all cells >= 0.60",
            "A4": (
                "Among cells where human gain is significantly negative (reward "
                "selection hurts humans), the judge gain is negative in >= 0.80 of "
                "them; if there are fewer than 5 such cells, A4 is reported as "
                "underpowered rather than passed"
            ),
            "decision": (
                "Accept the judge as the development proxy for generative-RL "
                "evaluation only if A1-A3 pass and A4 is not failed. Otherwise "
                "report failure and do not use the judge for headline claims."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"wrote {OUTPUT} sha256={digest}")
    for name, pool in pools.items():
        print(name, pool["segments_available"], len(pool["systems"]), pool["sampled_candidates"])


if __name__ == "__main__":
    main()
