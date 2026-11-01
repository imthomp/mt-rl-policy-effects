"""Shared WMT26 candidate-pool rules for policy and selection studies."""

from __future__ import annotations

from typing import Any


EXCLUDED_NON_MACHINE_SYSTEMS = frozenset({"Human (postediting)"})


def machine_only_rows(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    observed = {row["system_id"] for row in rows.values()}
    missing = EXCLUDED_NON_MACHINE_SYSTEMS - observed
    if missing:
        raise ValueError(f"Expected excluded systems are absent: {sorted(missing)}")
    return {
        key: row
        for key, row in rows.items()
        if row["system_id"] not in EXCLUDED_NON_MACHINE_SYSTEMS
    }
