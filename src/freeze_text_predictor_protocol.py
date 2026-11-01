"""Freeze the prospective text-policy predictor confirmation protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CALIBRATION_DOCUMENTS = 64
CONFIRMATION_REWARD_SEEDS = (20261203, 20261204, 20261205)
KL_GRID_NATS = (0.0, 0.1, 0.25, 0.5, 1.0)
PRIMARY_KL_NATS = 0.25
MAX_OPTIMIZATION_STEPS = 5000
LEARNING_RATE = 0.5
BEST_OF_N_GRID = (2, 4, 8)
PRIMARY_BEST_OF_N = 4


def build(root: Path) -> dict[str, Any]:
    manifest = json.loads(
        (root / "results/predictor_confirmation_manifest.json").read_text(encoding="utf-8")
    )
    development = json.loads(
        (root / "results/text_policy_predictor_development.json").read_text(encoding="utf-8")
    )
    budget = json.loads(
        (root / "results/text_policy_predictor_label_budget.json").read_text(encoding="utf-8")
    )
    if not manifest["status"].startswith("datasets_frozen"):
        raise ValueError("Confirmation data manifest is not frozen")
    if any(
        len(dataset["calibration_evaluation_split"]["calibration_segment_ids"])
        != CALIBRATION_DOCUMENTS
        for dataset in manifest["datasets"].values()
    ):
        raise ValueError("Manifest calibration budget differs from protocol")
    if development["frozen_confirmation_outcomes_accessed"] or budget[
        "frozen_confirmation_outcomes_accessed"
    ]:
        raise ValueError("Development analysis reports confirmation-outcome access")
    selected_budget = budget["budgets"][str(CALIBRATION_DOCUMENTS)]
    if selected_budget["decision_agreement_with_full_development_predictor"]["low"] < 1.0:
        raise ValueError("Selected label budget does not preserve all development decisions")
    if selected_budget["false_optimize_rate_on_development_failures"]["high"] > 0.0:
        raise ValueError("Selected label budget has development false optimizations")
    if selected_budget["optimize_rate_on_development_nonfailures"]["low"] < 1.0:
        raise ValueError("Selected label budget misses development nonfailures")
    return {
        "status": "protocol_frozen_confirmation_outcomes_sealed",
        "research_question": (
            "Can the sign of a pre-optimization first-order human/reward alignment screen "
            "choose between reward optimization and the initial policy on new reward seeds "
            "and document-disjoint human-rated candidate pools?"
        ),
        "development_evidence": {
            "source": "results/text_policy_predictor_development.json",
            "label_budget_source": "results/text_policy_predictor_label_budget.json",
            "reward_instances": development["reward_instance_count"],
            "reward_training_seeds": development["reward_seed_count"],
            "external_single_checkpoints": development["external_single_checkpoint_count"],
            "selected_budget_diagnostics": selected_budget,
            "status": "already viewed; used only to freeze this rule",
        },
        "confirmation_data": {
            "manifest": "results/predictor_confirmation_manifest.json",
            "primary": "wmt24_ja_zh",
            "secondary": "wmt23_zh_en",
            "calibration_documents_per_dataset": CALIBRATION_DOCUMENTS,
            "calibration_segments_per_document": 1,
            "evaluation": "all eligible segments in documents disjoint from calibration",
            "outcomes_at_freeze": "sealed",
        },
        "policy": {
            "representation": "frozen source-candidate XLM-R embeddings",
            "encoder_name": "FacebookAI/xlm-roberta-base",
            "encoder_revision": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
            "scorer": "one shared linear vector; no system identity or bias",
            "initial_policy": "uniform over the available training systems",
            "system_split": "exact train/heldout system lists in the frozen manifest",
        },
        "predictor": {
            "name": "initial text-policy human/reward gradient cosine",
            "reward_gradient_data": "all calibration-pool sources and frozen train systems; no human outcomes required",
            "human_gradient_data": "one frozen segment from each calibration document and frozen train systems",
            "reward_standardization": "mean zero and unit variance on the reward-gradient pool",
            "decision_rule": "optimize iff cosine > 0; ties and nonfinite values keep the initial policy",
            "threshold": 0.0,
            "theory": "the sign is the first-order predicted human-quality change under the reward update",
        },
        "optimization": {
            "estimator": "exact finite-action policy gradient",
            "learning_rate": LEARNING_RATE,
            "maximum_steps": MAX_OPTIMIZATION_STEPS,
            "pressure_measure": "mean KL(pi || uniform) in nats on the optimization pool",
            "pressure_grid_nats": list(KL_GRID_NATS),
            "primary_pressure_nats": PRIMARY_KL_NATS,
            "checkpoint_rule": "first checkpoint reaching each target; report unreachable targets",
        },
        "outcome": {
            "primary": "mean human-quality change from uniform on held-out documents crossed with held-out systems at the primary KL target",
            "failure": "primary human-quality change < 0",
            "secondary": [
                "human-quality curve over the full KL grid",
                "proxy-reward curve over the full KL grid",
                "decision regret against max(initial, optimized)",
                "false-optimize and false-keep rates",
            ],
            "checkpoint_selection_uses_human_outcomes": False,
        },
        "baselines": {
            "always_keep_initial": True,
            "always_optimize": True,
            "best_of_n_grid": list(BEST_OF_N_GRID),
            "best_of_n_primary": PRIMARY_BEST_OF_N,
            "best_of_n_rule": (
                "exact expectation over uniformly drawn candidate subsets on calibration data; "
                "optimize iff primary-N selected human quality exceeds uniform"
            ),
        },
        "reward_training": {
            "conditions": [
                "all_data",
                "diagnostic_filtered",
                "direction_matched_earlier_year",
                "global_random_2",
            ],
            "new_seeds": list(CONFIRMATION_REWARD_SEEDS),
            "external_positive_controls": [
                "wmt22-cometkiwi-da",
                "wmt23-cometkiwi-da-xl",
            ],
            "all_runs_reported": True,
        },
        "analysis": {
            "primary_unit": "reward checkpoint crossed with confirmation dataset",
            "paired_reporting": "show every row and pair decisions/outcomes within checkpoint and dataset",
            "uncertainty": "hierarchical bootstrap over reward-training seed and evaluation document; external controls reported separately",
            "success_criterion": (
                "the frozen sign screen has lower mean decision regret than both always-optimize "
                "and the frozen best-of-N screen on new learned-reward checkpoints"
            ),
            "no_post_outcome_changes": True,
        },
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["text_policy_predictor_protocol"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = build(root)
    output = root / "results/text_policy_predictor_protocol.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    merge_ssot(root / "results/paper_stats.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
