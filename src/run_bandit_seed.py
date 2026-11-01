#!/usr/bin/env python3
"""Full-horizon bandit for one reward seed."""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import run_contextual_bandit as rcb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probabilities-output", type=Path)
    args = parser.parse_args()
    output = args.output or REPO / f"results/bandit_seeds/seed_{args.seed}.json"
    probabilities = args.probabilities_output or (
        REPO / f"artifacts/reward_models/seeds/{args.seed}/probabilities.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{}\n")
    rcb.STEPS = 3200
    rcb.POLICY_SEEDS = rcb.POLICY_SEEDS[:1]
    sys.argv = [
        "seed_bandit",
        "--output", str(output),
        "--prediction-root", str(
            REPO / f"artifacts/reward_models/seeds/{args.seed}/predictions"
        ),
        "--probabilities-output", str(probabilities),
    ]
    rcb.main()


if __name__ == "__main__":
    main()
