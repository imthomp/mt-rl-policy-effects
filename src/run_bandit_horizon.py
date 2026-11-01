#!/usr/bin/env python3
"""Run the contextual bandit past its frozen horizon, unchanged otherwise, to a separate file."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_contextual_bandit as rcb

rcb.STEPS = 3200
rcb.POLICY_SEEDS = rcb.POLICY_SEEDS[:1]
sys.argv = ["run_bandit_horizon", "--output", str(ROOT / "results/bandit_horizon_raw.json"),
            "--probabilities-output", str(ROOT / "artifacts/bandit/horizon_probabilities.npz")]
rcb.main()
