"""Validation of thin sidecar source evidence against a gem5 checkout."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess

from .sidecar import Gem5Sidecar


@dataclass(frozen=True)
class SourceEvidence:
    """Resolved source reference with line locations and content digest."""

    ref_id: str
    path: str
    symbol_line: int
    snippet_lines: tuple[int, ...]
    sha256: str


def _line_number(text: str, needle: str, context: str) -> int:
    count = text.count(needle)
    if count != 1:
        raise ValueError(f"{context} must occur exactly once, found {count}")
    return text[: text.index(needle)].count("\n") + 1


def git_revision(repository: Path) -> str:
    """Return the checked-out git revision without modifying the repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_source_references(
    sidecar: Gem5Sidecar,
    gem5_root: Path,
    *,
    check_revision: bool = True,
) -> tuple[SourceEvidence, ...]:
    """Resolve every sidecar source reference against one gem5 checkout."""
    root = gem5_root.resolve()
    if check_revision:
        actual_revision = git_revision(root)
        if actual_revision != sidecar.gem5_revision:
            raise ValueError(
                f"gem5 revision mismatch: expected={sidecar.gem5_revision}, "
                f"actual={actual_revision}"
            )

    evidence: list[SourceEvidence] = []
    for ref_id, reference in sorted(sidecar.source_refs.items()):
        relative = Path(reference.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"source_refs.{ref_id}.path must stay within gem5 root")
        source_path = (root / relative).resolve()
        if not source_path.is_relative_to(root):
            raise ValueError(f"source_refs.{ref_id}.path escapes gem5 root")
        if not source_path.is_file():
            raise ValueError(f"source_refs.{ref_id}.path is not a file: {relative}")
        text = source_path.read_text(encoding="utf-8")
        symbol_line = _line_number(text, reference.symbol, f"{ref_id}.symbol")
        snippet_lines = tuple(
            _line_number(text, snippet, f"{ref_id}.snippet")
            for snippet in reference.snippets
        )
        evidence.append(
            SourceEvidence(
                ref_id=ref_id,
                path=reference.path,
                symbol_line=symbol_line,
                snippet_lines=snippet_lines,
                sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    return tuple(evidence)
