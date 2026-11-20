"""Small provenance helpers shared by reproducible experiment artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


_RUNTIME_SOURCE_PATHS = ("src", "scripts", "configs", "pyproject.toml")
_IGNORED_SOURCE_PARTS = {"__pycache__", ".pytest_cache"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def working_tree_sha256(root: str | Path) -> str:
    """Hash executable source/config content, including untracked files."""

    repository = Path(root).resolve()
    files: list[Path] = []
    for relative in _RUNTIME_SOURCE_PATHS:
        candidate = repository / relative
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_dir():
            files.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file()
                and not any(part in _IGNORED_SOURCE_PARTS for part in path.parts)
                and path.suffix not in {".pyc", ".pyo"}
            )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(repository).as_posix()):
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def git_source_revision(root: str | Path | None = None) -> dict[str, object]:
    repository = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        source_hash: str | None = working_tree_sha256(repository)
    except OSError:
        source_hash = None
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
        return {
            "commit": None,
            "tracked_dirty": None,
            "working_tree_sha256": source_hash,
        }
    return {
        "commit": commit,
        "tracked_dirty": bool(status.strip()),
        "working_tree_sha256": source_hash,
    }
