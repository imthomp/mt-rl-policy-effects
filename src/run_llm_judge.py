"""Run one frozen LLM MQM judge over the frozen validation sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from mqm_human_scores import error_penalty
from score_mqm_confirmation_candidates import load_candidates

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/llm_judge_protocol.json"
PROTOCOL_SHA256 = "172360f175fe106d44d4c666582e9a56a5eceb3ba6ff4d785b9618a0a9786013"
MQM_ROOT = Path.home() / "projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation"
SEVERITIES = {"minor", "major", "critical"}
AMENDMENT = ROOT / "results/llm_judge_protocol_amendment_1.json"
AMENDMENT_SHA256 = "1792c5b13514f0d4e9fef23e6baadc3dd1ca2415ebd72d6e6cd82e12c4875d87"


def parse_errors(text: str) -> list[dict[str, str]] | None:
    """Return the error list from the last JSON object containing "errors"."""
    for match in reversed(list(re.finditer(r"\{", text))):
        candidate = text[match.start():]
        try:
            obj, _ = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("errors"), list):
            errors = []
            for item in obj["errors"]:
                if not isinstance(item, dict):
                    return None
                severity = str(item.get("severity", "")).strip().casefold()
                if severity not in SEVERITIES:
                    return None
                errors.append(
                    {
                        "span": str(item.get("span", "")),
                        "category": str(item.get("category", "other")),
                        "severity": severity,
                    }
                )
            return errors
    return None


def judge_penalty(errors: list[dict[str, str]]) -> float:
    return sum(error_penalty(e["category"], e["severity"]) for e in errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--items", type=Path, default=None, help="frozen T4 item file")
    args = parser.parse_args()
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("LLM judge protocol differs from frozen hash")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    protocol = json.loads(PROTOCOL.read_text())
    model_name = protocol["judges"][args.judge]
    decoding = protocol["decoding"]

    items = []
    if args.items is not None:
        amendment = json.loads(AMENDMENT.read_text())
        if hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() != AMENDMENT_SHA256:
            raise ValueError("T4 amendment differs from frozen hash")
        if hashlib.sha256(args.items.read_bytes()).hexdigest() != amendment["items_sha256"]:
            raise ValueError("T4 items differ from frozen hash")
        for line in args.items.open(encoding="utf-8"):
            row = json.loads(line)
            pool = protocol["pools"][row["pool"]]
            row["prompt"] = protocol["prompt_template"].format(
                source_lang=pool["source_lang"],
                target_lang=pool["target_lang"],
                source=row["source"],
                hypothesis=row["hypothesis"],
            )
            row["system_id"] = f"{row['base_system']}::{row['perturbation']}"
            items.append(row)
    for pool_name, pool in ({} if args.items is not None else protocol["pools"]).items():
        _, rows = load_candidates(
            MQM_ROOT / pool["source"], ROOT / pool["manifest"], pool_name
        )
        sampled = set(pool["sampled_segments"])
        for row in rows:
            if row["segment_key"] not in sampled:
                continue
            prompt = protocol["prompt_template"].format(
                source_lang=pool["source_lang"],
                target_lang=pool["target_lang"],
                source=row["source"],
                hypothesis=row["hypothesis"],
            )
            items.append(
                {
                    "pool": pool_name,
                    "segment_key": row["segment_key"],
                    "system_id": row["system_id"],
                    "prompt": prompt,
                }
            )
        expected = pool["sampled_candidates"]
        observed = sum(1 for item in items if item["pool"] == pool_name)
        if observed != expected:
            raise ValueError(f"{pool_name}: expected {expected} candidates, got {observed}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=decoding["max_model_len"],
        trust_remote_code=False,
        seed=0,
    )
    params = SamplingParams(
        temperature=decoding["temperature"], max_tokens=decoding["max_tokens"]
    )
    conversations = [[{"role": "user", "content": item["prompt"]}] for item in items]
    outputs = llm.chat(conversations, params, use_tqdm=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    parsed_count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "record_type": "metadata",
                    "judge": args.judge,
                    "model": model_name,
                    "protocol_sha256": PROTOCOL_SHA256,
                    "candidates": len(items),
                },
                sort_keys=True,
            )
            + "\n"
        )
        for item, output in zip(items, outputs, strict=True):
            text = output.outputs[0].text
            errors = parse_errors(text)
            parsed_count += errors is not None
            handle.write(
                json.dumps(
                    {
                        "record_type": "candidate",
                        "pool": item["pool"],
                        "segment_key": item["segment_key"],
                        "system_id": item["system_id"],
                        "perturbation": item.get("perturbation"),
                        "raw_output": text,
                        "finish_reason": output.outputs[0].finish_reason,
                        "errors": errors,
                        "judge_quality": None if errors is None else -judge_penalty(errors),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    print(f"{args.judge}: parsed {parsed_count}/{len(items)}")


if __name__ == "__main__":
    main()
