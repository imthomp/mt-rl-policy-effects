"""Flag numeric literals in hand-authored LaTeX sources."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


NUMBER = re.compile(r"(?<![A-Za-z\\])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.IGNORECASE)
STRUCTURAL_EXCEPTIONS = (
    re.compile(r"\\documentclass\[11pt\]"),
    re.compile(r"\\usepackage\[utf8\]\{inputenc\}"),
    re.compile(r"\\cite[a-zA-Z*]*\{[^}]+\}"),
    # Mathematical zero vector in an analytic identity, not an empirical quantity.
    re.compile(r"\\mathbf\{0\}"),
)


def candidates(path: Path) -> list[str]:
    findings = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("%", 1)[0]
        stripped = line
        for pattern in STRUCTURAL_EXCEPTIONS:
            stripped = pattern.sub("", stripped)
        for match in NUMBER.finditer(stripped):
            findings.append(f"{path}:{line_number}: {match.group(0)}: {raw_line.strip()}")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    findings = [finding for path in args.paths for finding in candidates(path)]
    if findings:
        raise SystemExit("Candidate hardcoded numeric literals:\n" + "\n".join(findings))
    print("No candidate hardcoded numeric literals in hand-authored TeX.")


if __name__ == "__main__":
    main()
