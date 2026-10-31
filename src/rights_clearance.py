"""Require the project's documented rights-use decision before model use."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CLEARANCE_RECORD = Path(__file__).resolve().parents[1] / "docs/rights_clearance.json"


def require_rights_clearance() -> dict[str, Any]:
    if not CLEARANCE_RECORD.is_file():
        raise FileNotFoundError(f"rights-clearance record is missing: {CLEARANCE_RECORD}")
    record = json.loads(CLEARANCE_RECORD.read_text(encoding="utf-8"))
    if record.get("status") != "cleared":
        raise RuntimeError(
            "model use is paused until docs/rights_clearance.json records the project's rights-use decision"
        )
    required = ("scope", "reviewer_role", "decision_date", "decision_reference")
    missing = [field for field in required if not record.get(field)]
    if missing:
        raise ValueError(f"rights-clearance record lacks required fields: {missing}")
    return record


def clearance_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": record["status"],
        "reviewer_role": record["reviewer_role"],
        "decision_date": record["decision_date"],
        "decision_reference": record["decision_reference"],
    }
