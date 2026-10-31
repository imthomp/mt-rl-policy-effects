"""Repository revision metadata for reproducible model runs."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


BASELINE_COMMIT = "abf1ccc"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def repository_metadata() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "baseline_commit": BASELINE_COMMIT,
        "executing_commit": revision,
        # Keep the original keys for existing artifact readers.
        "research_snapshot_commit": BASELINE_COMMIT,
        "git_commit_sha": revision,
        "git_worktree_dirty": dirty,
    }
