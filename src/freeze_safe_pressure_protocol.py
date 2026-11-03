"""Freeze the prospective safe reward-optimization pressure protocol.

This file defines the study before a fresh confirmation dataset is selected.
It deliberately contains no reader for future human evaluation outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "1.0"
CALIBRATION_DOCUMENTS = 64
CALIBRATION_SEGMENTS_PER_DOCUMENT = 1
NORMALIZED_KL_GRID = (0.0, 0.025, 0.05, 0.10, 0.20, 0.35)
FIXED_PRESSURE_BASELINES = (0.05, 0.10, 0.20)
OPTIMIZATION_LEARNING_RATE = 0.5
MAXIMUM_OPTIMIZATION_STEPS = 20_000
CHECKPOINT_INTERVAL = 10
BOOTSTRAP_SEED = 20270116
BOOTSTRAP_RESAMPLES = 10_000
INTERVAL_LEVEL = 0.95
REWARD_TRAINING_SEEDS = (20270111, 20270112, 20270113, 20270114, 20270115)


def build(root: Path) -> dict[str, Any]:
    prior = json.loads(
        (root / "results/text_policy_predictor_confirmation_v2.json").read_text(
            encoding="utf-8"
        )
    )
    if prior["status"] != "prospective_confirmation_complete":
        raise ValueError("The preceding prospective confirmation is incomplete")
    if not prior["evaluation_outcomes_accessed"]:
        raise ValueError("Prior outcomes must be classified as development evidence")

    excluded = {
        "wmt25_released_mqm_candidate_pools": (
            "Used in reward-model and best-of-N development"
        ),
        "wmt26_zh_ja_cesa": "Used in reward-model and bandit development",
        "wmt23_zh_en_mqm": "Used in the completed prospective sign-screen test",
        "wmt24_ja_zh_mqm": "Used in the completed prospective sign-screen test",
    }

    return {
        "status": "design_frozen_fresh_confirmation_manifest_pending",
        "protocol_version": PROTOCOL_VERSION,
        "frozen_before": {
            "fresh_confirmation_dataset_selected": True,
            "fresh_confirmation_human_outcomes_accessed": True,
            "fresh_confirmation_reward_scores_computed": True,
            "fresh_confirmation_policy_artifacts_computed": True,
        },
        "research_question": (
            "Can a bounded calibration audit choose a reward-optimization pressure "
            "that reduces held-out human-quality decision regret relative to fixed "
            "pressure, best-of-N, always-optimize, and always-keep baselines?"
        ),
        "development_evidence": {
            "completed_sign_screen_result": (
                "results/text_policy_predictor_confirmation_v2.json"
            ),
            "classification": (
                "All existing outcomes are development evidence and may motivate this "
                "design, but none may enter the future confirmation endpoint."
            ),
            "observed_design_fact": (
                "The completed sign screen discarded beneficial low-pressure updates; "
                "the new decision therefore selects pressure rather than only optimize/keep."
            ),
        },
        "confirmation_data": {
            "manifest_status": "not yet created",
            "manifest_freeze_rule": (
                "Create and hash exact dataset revision, documents, segments, candidate "
                "systems, reward checkpoints, and splits using metadata and candidate "
                "coverage only, before reading any human score."
            ),
            "eligibility": [
                "publicly releasable source and candidate text",
                "human scores available at candidate level but unopened at selection time",
                "document identifiers sufficient for disjoint calibration and evaluation",
                "enough machine systems for disjoint train-system and heldout-system sets",
                "rectangular candidate coverage after prespecified non-machine exclusions",
                "no dataset or direction listed in excluded_previously_examined_pools",
            ],
            "excluded_previously_examined_pools": excluded,
            "dataset_selection_failure_rule": (
                "If no eligible fresh pool exists, stop prospective confirmation and "
                "report the feasibility failure; do not relabel a viewed pool as fresh."
            ),
            "split": {
                "calibration_documents": CALIBRATION_DOCUMENTS,
                "calibration_segments_per_document": CALIBRATION_SEGMENTS_PER_DOCUMENT,
                "calibration_sampling": (
                    "deterministic hash-ranked document sample stratified by declared domain; "
                    "one deterministic hash-ranked eligible segment per selected document"
                ),
                "evaluation": "all eligible segments in all remaining documents",
                "system_split": (
                    "deterministic hash-ranked balanced split fixed in the manifest; policy "
                    "receives source-candidate text and no system identity"
                ),
            },
            "outcome_access_order": [
                "freeze and hash dataset/reward/split manifest",
                "compute label-free audits and all policy trajectories",
                "validate and hash every policy artifact",
                "open calibration human labels and freeze pressure decisions",
                "open evaluation human labels exactly once",
            ],
        },
        "reward_models": {
            "learned_conditions": [
                "all_available_DA",
                "metric_annotation_disagreement_slices_removed",
                "matched_random_slices_removed",
            ],
            "independent_training_seeds": list(REWARD_TRAINING_SEEDS),
            "external_controls": (
                "Freeze exact public checkpoint revisions in the future manifest; report "
                "them separately because checkpoints are not reward-training replicates."
            ),
            "training_control": (
                "Architecture, initialization distribution, optimizer, budget, and score "
                "calibration are identical across learned conditions."
            ),
            "all_runs_rule": "Retain and report every scheduled reward run.",
        },
        "policy": {
            "representation": "frozen source-candidate XLM-R embeddings",
            "encoder_name": "FacebookAI/xlm-roberta-base",
            "encoder_revision": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
            "scorer": "one shared linear vector; no system identity and no bias",
            "initial_policy": "uniform over train-system candidates",
            "optimizer": "exact finite-action policy gradient",
            "reward_standardization": (
                "mean zero and unit variance on the unlabeled optimization pool"
            ),
            "learning_rate": OPTIMIZATION_LEARNING_RATE,
            "maximum_steps": MAXIMUM_OPTIMIZATION_STEPS,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
        },
        "pressure": {
            "measure": (
                "mean KL(pi || uniform) divided by log(number of available actions), "
                "computed on the unlabeled optimization pool"
            ),
            "interpretation": "zero is uniform and one is maximum concentration",
            "targets": list(NORMALIZED_KL_GRID),
            "checkpoint_rule": (
                "first recorded checkpoint reaching each target; preserve the final "
                "checkpoint and mark every unreached target unavailable"
            ),
            "unreachable_rule": (
                "A selector may choose only a target reached by every reward checkpoint "
                "being compared in its analysis stratum. No outcome is imputed."
            ),
        },
        "label_free_audits": {
            "role": (
                "Secondary predictors and mechanism diagnostics. They do not change the "
                "primary pressure decision in protocol version 1.0."
            ),
            "features": {
                "top_system_preference_margin": (
                    "largest system mean standardized reward minus the grand mean"
                ),
                "between_system_separation": (
                    "fraction of residual standardized reward variance attributable to system"
                ),
                "initial_gradient_norm": "Euclidean norm of the exact reward gradient",
                "initial_concentration_rate": (
                    "first-order normalized-KL change under the frozen update"
                ),
                "reward_tail_gap": (
                    "mean within-source gap between the largest and second-largest reward"
                ),
                "reward_seed_disagreement": (
                    "mean within-source variance of standardized scores across independent "
                    "reward seeds in the same training condition"
                ),
            },
            "analysis_rule": (
                "Report associations with oracle safe pressure with leave-one-dataset-out "
                "estimates; do not promote a feature to a confirmation decision after "
                "evaluation outcomes are opened."
            ),
        },
        "primary_selector": {
            "name": "bounded human-calibration pressure argmax",
            "human_feedback": (
                "candidate human scores for the frozen calibration segment from each "
                "calibration document and train-system candidates only"
            ),
            "calibration_score": (
                "within-segment candidate human scores standardized by the segment standard "
                "deviation, then expected under each frozen policy checkpoint and averaged "
                "equally over calibration documents"
            ),
            "decision_rule": (
                "Among jointly reachable pressure targets, choose the target with the largest "
                "mean calibration human-quality change from uniform; ties within exact "
                "floating-point equality choose the smaller pressure. The zero-pressure "
                "action is always available."
            ),
            "no_evaluation_labels": True,
            "decision_artifact_rule": (
                "Write and hash every per-reward pressure decision before evaluation labels "
                "are opened."
            ),
        },
        "secondary_selectors": {
            "pessimistic_calibration": (
                "choose the pressure with the largest document-bootstrap lower confidence "
                "bound on calibration human change; choose zero unless that bound is positive"
            ),
            "first_order_sign": (
                "completed sign rule applied without modification as a historical baseline"
            ),
            "best_of_n": (
                "exact expected calibration human change for uniformly sampled candidate "
                "subsets; choose its best prespecified N or keep uniform"
            ),
        },
        "baselines": {
            "always_keep_initial": True,
            "fixed_normalized_kl_targets": list(FIXED_PRESSURE_BASELINES),
            "always_optimize": (
                "largest jointly reachable frozen pressure target, defined without outcomes"
            ),
            "best_of_n_subset_sizes": [2, 4, 8],
            "human_oracle": (
                "evaluation-only ceiling choosing the pressure with greatest held-out human "
                "quality; never available to a decision method"
            ),
        },
        "outcomes": {
            "primary": (
                "held-out decision regret: oracle held-out human quality minus human quality "
                "at the selector's pressure, averaged over learned reward checkpoints"
            ),
            "human_quality": (
                "candidate-level released human score aggregated by its official protocol; "
                "larger-is-better direction fixed in the dataset manifest"
            ),
            "secondary": [
                "held-out human-quality change from uniform",
                "standardized proxy-reward gain from uniform",
                "selected raw and normalized KL pressure",
                "harmful optimization rate",
                "missed-benefit regret",
                "worst observed per-checkpoint regret",
                "unreachable-target rate",
                "calibration label cost",
            ],
            "reporting": (
                "Show complete pressure curves and every reward checkpoint, including failed, "
                "degenerate, and unreachable runs."
            ),
        },
        "analysis": {
            "primary_unit": "independently trained reward checkpoint crossed with dataset",
            "policy_seed_scope": (
                "Policy seeds, if sampled REINFORCE is added, measure optimizer variability "
                "and never replace reward-training seeds."
            ),
            "uncertainty": (
                "hierarchical bootstrap resampling reward-training seed and evaluation "
                "document; external controls reported separately"
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "interval_level": INTERVAL_LEVEL,
            "primary_contrasts": [
                "primary selector minus each fixed-pressure baseline",
                "primary selector minus always optimize",
                "primary selector minus always keep",
                "primary selector minus best-of-N",
            ],
            "success_criterion": (
                "The primary selector has lower mean held-out decision regret than every "
                "fixed-pressure baseline and best-of-N, and its interval excludes zero for "
                "the contrast with the strongest non-oracle baseline."
            ),
            "inconclusive_rule": (
                "If any primary comparison is undefined because a pressure target is "
                "unreachable or an analysis stratum is absent, report the criterion as "
                "inconclusive; do not impute, replace the endpoint, or tune the grid."
            ),
        },
        "claim_boundary": (
            "The study concerns pressure selection for fixed candidate-pool MT policies. "
            "It does not establish safety for generative RL or prove that a removed DA slice "
            "contains incorrect individual annotations."
        ),
        "next_freeze": (
            "An outcome-blind data audit must produce the exact confirmation manifest and "
            "hash this protocol before any fresh candidate human score is read."
        ),
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["safe_pressure_protocol"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = build(root)
    output = root / "results/safe_pressure_protocol.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite the frozen safe-pressure protocol")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    merge_ssot(root / "results/paper_stats.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
