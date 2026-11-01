"""Build the tokenization note's SSOT section and generated LaTeX macros."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from exact_same_string_harness import enumerate_outcomes, report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_absolute_difference(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def display_path(path: Path) -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def table_row_trace(result: dict[str, Any]) -> dict[str, dict[str, str]]:
    synthetic = result["synthetic"]
    calibration = result["tatoeba_calibration"]
    synthetic_rows = {
        "response_mean_phi": "synthetic.q_sampler_sensitivity[*].response_mean_phi",
        "global_token_mean_phi": "synthetic.q_sampler_sensitivity[*].global_token_mean_phi",
    }
    calibration_rows = {
        "tokens_per_word": "tatoeba_calibration.fertility",
        "chrf_reward": "tatoeba_calibration.chrf_reward",
        "corpus_bleu_points": "tatoeba_calibration.corpus_bleu_points",
        "actions_including_eos": "tatoeba_calibration.actions_including_eos",
        "decoded_word_count": "tatoeba_calibration.decoded_word_count",
    }
    calibration_source_keys = {
        "tokens_per_word": "fertility",
        "chrf_reward": "chrf_reward",
        "corpus_bleu_points": "corpus_bleu_points",
        "actions_including_eos": "actions_including_eos",
        "decoded_word_count": "decoded_word_count",
    }
    if len(synthetic["q_sampler_sensitivity"]) != 2:
        raise AssertionError("Synthetic table requires exactly two recorded q settings")
    for row_name, key in calibration_source_keys.items():
        row = calibration[key]
        if "mean_length_tilt_minus_control" not in row:
            raise AssertionError(f"Missing table mean source: {row_name}")
        bounds = row.get("paired_source_bootstrap_95_interval", {})
        if not {"lower_95", "upper_95"} <= set(bounds):
            raise AssertionError(f"Missing table interval source: {row_name}")
    return {"synthetic_table": synthetic_rows, "calibration_table": calibration_rows}


def build_result(combined_path: Path, initial_path: Path) -> dict[str, Any]:
    synthetic = report()
    calibration = json.loads(combined_path.read_text(encoding="utf-8"))
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    q_low, q_high = synthetic["q_sampler_sensitivity"]
    mc_response = synthetic["monte_carlo_batch_gradients"]["response_mean"]
    mc_token = synthetic["monte_carlo_batch_gradients"]["global_token_mean"]
    paired = calibration["paired_differences"]
    decisions = calibration["screening_decisions"]
    result = {
        "status": "complete",
        "scope": "same-string estimator validation and NLLB length-tilt feasibility; no RL training effect",
        "synthetic": {
            **synthetic,
            "outcome_count": len(enumerate_outcomes(tuple(synthetic["theta_logits"]), synthetic["phi_logit"])),
            "response_theta_max_abs_difference_between_q_settings": maximum_absolute_difference(
                q_low["response_mean_theta"], q_high["response_mean_theta"]
            ),
            "monte_carlo_response_theta_max_abs_error": maximum_absolute_difference(
                mc_response[0], synthetic["exact_batch_response_mean_theta"]
            ),
            "monte_carlo_response_phi_abs_error": abs(
                mc_response[1] - synthetic["exact_response_mean_phi"]
            ),
            "monte_carlo_global_token_theta_max_abs_error": maximum_absolute_difference(
                mc_token[0], synthetic["exact_batch_token_mean_theta"]
            ),
            "monte_carlo_global_token_phi_abs_error": abs(
                mc_token[1] - synthetic["exact_batch_token_mean_phi"]
            ),
        },
        "tatoeba_calibration": {
            "combined_artifact": display_path(combined_path),
            "combined_artifact_sha256": sha256_file(combined_path),
            "initial_artifact": display_path(initial_path),
            "initial_artifact_sha256": sha256_file(initial_path),
            "model_id": initial["model_id"],
            "model_snapshot_hash": initial["model_snapshot_hash"],
            "source_count": calibration["source_count"],
            "generated_output_count": decisions["validity_counts"]["generated_output_count"],
            "rollout_seed_count": len(calibration["rollout_seeds"]),
            "tilt_beta": initial["tilt_beta"],
            "tilt_temperature": initial["tilt_temperature"],
            "matched_temperature": initial["matched_temperature"],
            "entropy_match_tolerance_nats": initial["entropy_match_tolerance_nats"],
            "entropy_residual_nats": initial["matched_temperature_entropy_residual_nats"],
            "entropy_match_pass": initial["entropy_match_pass"],
            "fertility": paired["fertility_tokens_per_word"],
            "chrf_reward": paired["chrF++_reward"],
            "corpus_bleu_points": paired["corpus_BLEU_points"],
            "actions_including_eos": paired["actions_including_eos"],
            "decoded_word_count": paired["decoded_word_count"],
            "validity_counts": decisions["validity_counts"],
            "validity_rates": decisions["validity_rates"],
            "quality_decisions": decisions["quality"],
            "overall_screen": decisions["overall_calibration_screen"],
        },
    }
    result["table_row_trace"] = table_row_trace(result)
    return result


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["tokenization_paper"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def signed(value: float, decimals: int) -> str:
    return f"{value:+.{decimals}f}"


def interval(metric: dict[str, Any]) -> tuple[float, float]:
    bounds = metric["paired_source_bootstrap_95_interval"]
    return bounds["lower_95"], bounds["upper_95"]


def render_macros(path: Path, result: dict[str, Any]) -> None:
    synthetic = result["synthetic"]
    calibration = result["tatoeba_calibration"]
    q_low, q_high = synthetic["q_sampler_sensitivity"]
    fertility_low, fertility_high = interval(calibration["fertility"])
    chrf_low, chrf_high = interval(calibration["chrf_reward"])
    bleu_low, bleu_high = interval(calibration["corpus_bleu_points"])
    action_low, action_high = interval(calibration["actions_including_eos"])
    word_low, word_high = interval(calibration["decoded_word_count"])
    macros = {
        "ModelName": (
            calibration["model_id"]
            .replace("facebook/", "")
            .replace("nllb-", "NLLB-")
            .replace("-distilled-", " distilled ")
        ),
        "SyntheticStringCount": str(len(synthetic["strings"])),
        "SyntheticOutcomeCount": str(synthetic["outcome_count"]),
        "SyntheticBatchSize": str(synthetic["batch_size"]),
        "SyntheticMCBatches": f"{synthetic['batches']:,}",
        "QLow": f"{q_low['q_short']:.2f}",
        "QHigh": f"{q_high['q_short']:.2f}",
        "ResponsePhiLow": f"{q_low['response_mean_phi']:.1f}",
        "ResponsePhiHigh": f"{q_high['response_mean_phi']:.1f}",
        "GlobalPhiLow": f"{q_low['global_token_mean_phi']:.5f}",
        "GlobalPhiHigh": f"{q_high['global_token_mean_phi']:.5f}",
        "ResponseThetaMaxDiff": f"{synthetic['response_theta_max_abs_difference_between_q_settings']:.2e}",
        "MCResponseThetaError": f"{synthetic['monte_carlo_response_theta_max_abs_error']:.2e}",
        "MCResponsePhiError": f"{synthetic['monte_carlo_response_phi_abs_error']:.2e}",
        "MCGlobalThetaError": f"{synthetic['monte_carlo_global_token_theta_max_abs_error']:.2e}",
        "MCGlobalPhiError": f"{synthetic['monte_carlo_global_token_phi_abs_error']:.2e}",
        "TatoebaSources": f"{calibration['source_count']:,}",
        "TatoebaOutputs": f"{calibration['generated_output_count']:,}",
        "CalibrationReplicates": str(calibration["rollout_seed_count"]),
        "TiltBeta": f"{calibration['tilt_beta']:.2f}",
        "TiltTemperature": f"{calibration['tilt_temperature']:.2f}",
        "MatchedTemperature": f"{calibration['matched_temperature']:.2f}",
        "EntropyTolerance": f"{calibration['entropy_match_tolerance_nats']:.2f}",
        "EntropyResidual": signed(calibration["entropy_residual_nats"], 4),
        "FertilityDelta": signed(calibration["fertility"]["mean_length_tilt_minus_control"], 3),
        "FertilityLow": signed(fertility_low, 3),
        "FertilityHigh": signed(fertility_high, 3),
        "ChrfDelta": signed(calibration["chrf_reward"]["mean_length_tilt_minus_control"], 3),
        "ChrfLow": signed(chrf_low, 3),
        "ChrfHigh": signed(chrf_high, 3),
        "BleuDelta": signed(calibration["corpus_bleu_points"]["mean_length_tilt_minus_control"], 2),
        "BleuLow": signed(bleu_low, 2),
        "BleuHigh": signed(bleu_high, 2),
        "ActionDelta": signed(calibration["actions_including_eos"]["mean_length_tilt_minus_control"], 3),
        "ActionLow": signed(action_low, 3),
        "ActionHigh": signed(action_high, 3),
        "WordDelta": signed(calibration["decoded_word_count"]["mean_length_tilt_minus_control"], 3),
        "WordLow": signed(word_low, 3),
        "WordHigh": signed(word_high, 3),
        "EOSCount": f"{calibration['validity_counts']['eos_output_count']:,}",
        "TruncationCount": f"{calibration['validity_counts']['truncated_output_count']:,}",
        "EmptyCount": f"{calibration['validity_counts']['empty_output_count']:,}",
        "SpecialTokenCount": f"{calibration['validity_counts']['non_eos_special_token_output_count']:,}",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated from results/paper_stats.json; do not edit."]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combined",
        type=Path,
        default=root / "artifacts/diagnostics/tatoeba_policy_calibration_combined_14941363_14941543.json",
    )
    parser.add_argument(
        "--initial",
        type=Path,
        default=root / "artifacts/diagnostics/tatoeba_policy_calibration_14941363.json",
    )
    parser.add_argument("--ssot", type=Path, default=root / "results/paper_stats.json")
    parser.add_argument(
        "--macros",
        type=Path,
        default=root / "paper/tokenization_acl/generated_stats.tex",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_result(args.combined, args.initial)
    merge_ssot(args.ssot, result)
    render_macros(args.macros, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
