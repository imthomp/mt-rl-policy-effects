"""Build paper-facing statistics and macros for the AmanaMT-derived study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONDITIONS = (
    "all_data",
    "diagnostic_filtered",
    "global_random_2",
    "direction_matched_earlier_year",
)


def signed(value: float, decimals: int) -> str:
    return f"{value:+.{decimals}f}"


def condition_label(condition: str) -> str:
    return {
        "all_data": "All data",
        "diagnostic_filtered": "Disagreement-filtered",
        "global_random_2": "Size-matched random",
        "direction_matched_earlier_year": "Direction-matched earlier year",
    }[condition]


def build_result(root: dict[str, Any]) -> dict[str, Any]:
    diagnostics = root["reward_da_slice_diagnostics"]
    training = root["reward_training_conditions"]
    development = root["reward_model_development"]
    wmt25 = root["wmt25_best_of_n"]
    wmt26 = root["wmt26_best_of_n"]
    bandit = root["bandit_mechanism"]
    bandit_run = root["contextual_bandit"]

    wmt25_first = str(wmt25["pool_sizes"][0])
    wmt25_full = str(wmt25["candidate_pool_size"])
    wmt26_first = str(wmt26["pool_sizes"][0])
    wmt26_full = str(wmt26["candidate_pool_size"])

    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        w25 = wmt25["selection_results"]["combined"]["conditions"][condition]
        w26 = wmt26["conditions"][condition]
        exact = bandit["conditions"][condition]["exact_policy"]
        top = bandit["conditions"][condition]["top_mean_action_probabilities"][0]
        conditions[condition] = {
            "label": condition_label(condition),
            "excluded_slice_count": len(training["conditions"][condition]["excluded_slices"]),
            "removed_source_rows": training["conditions"][condition]["removed_rows"],
            "development_spearman": development["conditions"][condition]["dev_spearman"],
            "wmt25_full_human_mqm": w25[wmt25_full],
            "wmt25_change_first_to_full": (
                w25[wmt25_full]["mean"] - w25[wmt25_first]["mean"]
            ),
            "wmt26_full_human_cesa": w26[wmt26_full],
            "wmt26_change_first_to_full": (
                w26[wmt26_full]["mean"] - w26[wmt26_first]["mean"]
            ),
            "bandit_exact": exact,
            "bandit_top_action": top,
        }

    wmt25_contrasts = wmt25["selection_results"]["combined"]["contrasts"]
    wmt26_contrasts = wmt26["contrasts"]
    targeted_contrasts = {}
    for comparator in ("all_data", "global_random_2", "direction_matched_earlier_year"):
        key = f"diagnostic_filtered_minus_{comparator}"
        targeted_contrasts[comparator] = {
            "wmt25_full": wmt25_contrasts[key][wmt25_full],
            "wmt26_full": wmt26_contrasts[key][wmt26_full],
        }

    result = {
        "status": "initial_manuscript_draft",
        "claim_boundary": (
            "Reward optimization is sensitive to reward-training composition; "
            "metric-annotation disagreement filtering is not established as superior to matched removal."
        ),
        "diagnostic": {
            "definition": "Spearman(COMET,BLEURT) minus Spearman(COMET,DA)",
            "slice_count": diagnostics["slice_count"],
            "flagged_slice_count": diagnostics["flagged_slice_count"],
            "inconclusive_slice_count": diagnostics["inconclusive_slice_count"],
            "sample_rows": diagnostics["comet_metadata"]["total_rows"],
            "threshold": diagnostics["threshold"],
            "bootstrap_resamples": diagnostics["bootstrap_resamples"],
        },
        "reward_training": {
            "model_name": development["shared_configuration"]["model_name"],
            "train_rows_per_condition": development["shared_configuration"]["train_rows"],
            "development_rows": development["shared_configuration"]["dev_rows"],
            "steps": development["shared_configuration"]["steps"],
            "reward_checkpoint_seeds_per_condition": 1,
        },
        "wmt25": {
            "candidate_pool_size": wmt25["candidate_pool_size"],
            "segments": wmt25["selection_results"]["combined"]["conditions"]["all_data"][wmt25_full]["segments"],
            "permutations": wmt25["permutations"],
            "human_endpoint": "MQM score; higher is better",
        },
        "wmt26": {
            "candidate_pool_size": wmt26["candidate_pool_size"],
            "segments": wmt26["conditions"]["all_data"][wmt26_full]["segments"],
            "permutations": wmt26["permutations"],
            "human_endpoint": "cESA; higher is better",
            "excluded_non_machine_systems": wmt26["excluded_non_machine_systems"],
        },
        "bandit": {
            "contexts": bandit_run["environment"]["contexts"],
            "actions": bandit_run["environment"]["actions"],
            "train_contexts": bandit_run["environment"]["train_contexts"],
            "test_contexts": bandit_run["environment"]["test_contexts"],
            "test_documents": bandit["uniform_human"]["document_clusters"],
            "updates": bandit_run["protocol"]["steps"],
            "policy_seeds": len(bandit_run["protocol"]["policy_seeds"]),
            "uniform_human": bandit["uniform_human"],
            "independent_reconstruction": bandit[
                "independent_reconstruction_matches_recorded_metrics"
            ],
        },
        "conditions": conditions,
        "targeted_contrasts": targeted_contrasts,
    }
    result["table_row_trace"] = validate_table_rows(result)
    return result


def validate_table_rows(result: dict[str, Any]) -> dict[str, dict[str, str]]:
    traces = {
        "training_conditions": {},
        "best_of_n": {},
        "bandit": {},
        "targeted_contrasts": {},
    }
    for condition in CONDITIONS:
        row = result["conditions"][condition]
        required = {
            "excluded_slice_count",
            "removed_source_rows",
            "development_spearman",
            "wmt25_full_human_mqm",
            "wmt26_full_human_cesa",
            "bandit_exact",
        }
        if not required <= set(row):
            raise AssertionError(f"Incomplete paper row: {condition}")
        traces["training_conditions"][condition] = (
            f"reward_training_conditions.conditions.{condition}; "
            f"reward_model_development.conditions.{condition}"
        )
        traces["best_of_n"][condition] = (
            f"wmt25_best_of_n.selection_results.combined.conditions.{condition}; "
            f"wmt26_best_of_n.conditions.{condition}"
        )
        traces["bandit"][condition] = f"bandit_mechanism.conditions.{condition}.exact_policy"
    for comparator, row in result["targeted_contrasts"].items():
        for endpoint in ("wmt25_full", "wmt26_full"):
            if not {"mean", "ci_low", "ci_high", "segments"} <= set(row[endpoint]):
                raise AssertionError(f"Incomplete contrast row: {comparator}.{endpoint}")
        traces["targeted_contrasts"][comparator] = (
            f"wmt25_best_of_n...diagnostic_filtered_minus_{comparator}; "
            f"wmt26_best_of_n...diagnostic_filtered_minus_{comparator}"
        )
    return traces


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8"))
    root["reward_overoptimization_paper"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def macro_name(condition: str) -> str:
    return {
        "all_data": "All",
        "diagnostic_filtered": "Filtered",
        "global_random_2": "Random",
        "direction_matched_earlier_year": "Direction",
    }[condition]


def render_macros(path: Path, result: dict[str, Any]) -> None:
    diagnostic = result["diagnostic"]
    reward = result["reward_training"]
    bandit = result["bandit"]
    macros: dict[str, str] = {
        "DiagnosticSlices": f"{diagnostic['slice_count']:,}",
        "DiagnosticFlagged": f"{diagnostic['flagged_slice_count']:,}",
        "DiagnosticInconclusive": f"{diagnostic['inconclusive_slice_count']:,}",
        "DiagnosticRows": f"{diagnostic['sample_rows']:,}",
        "DiagnosticThreshold": f"{diagnostic['threshold']:.2f}",
        "RewardModelName": reward["model_name"].replace("FacebookAI/", ""),
        "RewardTrainRows": f"{reward['train_rows_per_condition']:,}",
        "RewardDevRows": f"{reward['development_rows']:,}",
        "RewardSteps": f"{reward['steps']:,}",
        "RewardCheckpointSeeds": str(reward["reward_checkpoint_seeds_per_condition"]),
        "PrimaryPool": str(result["wmt25"]["candidate_pool_size"]),
        "PrimarySegments": f"{result['wmt25']['segments']:,}",
        "ExternalPool": str(result["wmt26"]["candidate_pool_size"]),
        "ExternalSegments": f"{result['wmt26']['segments']:,}",
        "PoolPermutations": f"{result['wmt25']['permutations']:,}",
        "BanditContexts": f"{bandit['contexts']:,}",
        "BanditActions": f"{bandit['actions']:,}",
        "BanditTrainContexts": f"{bandit['train_contexts']:,}",
        "BanditTestContexts": f"{bandit['test_contexts']:,}",
        "BanditTestDocuments": f"{bandit['test_documents']:,}",
        "BanditUpdates": f"{bandit['updates']:,}",
        "BanditSeeds": str(bandit["policy_seeds"]),
        "UniformCesa": f"{bandit['uniform_human']['mean']:.2f}",
        "WMTPrimaryName": "WMT25",
        "WMTExternalName": "WMT26",
    }
    for condition, row in result["conditions"].items():
        name = macro_name(condition)
        macros[f"{name}ExcludedSlices"] = f"{row['excluded_slice_count']:,}"
        macros[f"{name}RemovedRows"] = f"{row['removed_source_rows']:,}"
        macros[f"{name}DevSpearman"] = f"{row['development_spearman']:.3f}"
        macros[f"{name}PrimaryFull"] = f"{row['wmt25_full_human_mqm']['mean']:.2f}"
        macros[f"{name}PrimaryChange"] = signed(row["wmt25_change_first_to_full"], 2)
        macros[f"{name}ExternalFull"] = f"{row['wmt26_full_human_cesa']['mean']:.2f}"
        macros[f"{name}ExternalChange"] = signed(row["wmt26_change_first_to_full"], 2)
        change = row["bandit_exact"]["human_change_from_uniform"]
        macros[f"{name}BanditCesa"] = f"{row['bandit_exact']['human_cesa']:.2f}"
        macros[f"{name}BanditChange"] = signed(change["mean"], 2)
        macros[f"{name}BanditLow"] = signed(change["ci_low"], 2)
        macros[f"{name}BanditHigh"] = signed(change["ci_high"], 2)
        macros[f"{name}TopProbability"] = f"{row['bandit_top_action']['probability']:.3f}"
    for comparator, row in result["targeted_contrasts"].items():
        name = macro_name(comparator)
        for endpoint, prefix in (("wmt25_full", "Primary"), ("wmt26_full", "External")):
            contrast = row[endpoint]
            macros[f"FilteredMinus{name}{prefix}"] = signed(contrast["mean"], 2)
            macros[f"FilteredMinus{name}{prefix}Low"] = signed(contrast["ci_low"], 2)
            macros[f"FilteredMinus{name}{prefix}High"] = signed(contrast["ci_high"], 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated from results/paper_stats.json; do not edit."]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssot", type=Path, default=root / "results/paper_stats.json")
    parser.add_argument(
        "--macros",
        type=Path,
        default=root / "paper/reward_overoptimization_acl/generated_stats.tex",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = json.loads(args.ssot.read_text(encoding="utf-8"))
    result = build_result(root)
    merge_ssot(args.ssot, result)
    render_macros(args.macros, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
