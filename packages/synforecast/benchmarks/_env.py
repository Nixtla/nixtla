"""Record the execution environment alongside benchmark results.

Benchmark numbers are only reproducible relative to the environment that
produced them; every script that saves results embeds this metadata.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

_PACKAGES = (
    "synforecast",
    "numpy",
    "pandas",
    "polars",
    "scipy",
    "scikit-learn",
    "statsforecast",
    "mlforecast",
    "neuralforecast",
    "utilsforecast",
    "datasetsforecast",
    "torch",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _git_metadata() -> dict:
    """Identify the repository state used to produce a benchmark."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": None,
            "git_dirty": None,
            "git_diff_sha256": None,
        }

    untracked = sorted(path for path in untracked if path)
    state_hash = hashlib.sha256(diff)
    for raw_path in untracked:
        state_hash.update(b"\0untracked\0")
        state_hash.update(raw_path)
        try:
            state_hash.update((_REPOSITORY_ROOT / os.fsdecode(raw_path)).read_bytes())
        except OSError:
            state_hash.update(b"\0unreadable")

    return {
        "git_commit": commit,
        "git_dirty": bool(diff or untracked),
        "git_diff_sha256": state_hash.hexdigest(),
    }


def environment_metadata() -> dict:
    """Versions, hardware, and the exact command that produced the results."""
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    environment = {
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "package_versions": versions,
    }
    environment.update(_git_metadata())
    return environment
