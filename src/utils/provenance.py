from __future__ import annotations

import subprocess
from pathlib import Path


def _run_git(args, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip()


def git_provenance(repo_root: str | Path | None = None) -> dict:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    commit = _run_git(["rev-parse", "HEAD"], root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    status = _run_git(["status", "--porcelain"], root)

    if commit is None:
        return {
            "git_commit": "unknown",
            "git_branch": "unknown",
            "git_dirty": "unknown",
            "git_status": "unknown",
        }

    dirty = bool(status)
    return {
        "git_commit": commit,
        "git_branch": branch or "unknown",
        "git_dirty": dirty,
        "git_status": "dirty" if dirty else "clean",
    }
