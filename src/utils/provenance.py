"""Small provenance helpers shared by reproducible experiment artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_source_revision(root: str | Path | None = None) -> dict[str, object]:
    repository = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tracked_dirty": None}
    return {"commit": commit, "tracked_dirty": bool(status.strip())}
