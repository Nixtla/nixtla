"""Release metadata consistency checks."""

import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).parents[1]


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_release_versions_are_consistent() -> None:
    """Keep Python, Rust, citation, and changelog versions synchronized."""
    version = _toml(ROOT / "pyproject.toml")["project"]["version"]
    cargo_version = _toml(ROOT / "rust" / "Cargo.toml")["package"]["version"]
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    citation_match = re.search(r"^version:\s*[\"']?([^\"'\s]+)", citation, re.MULTILINE)

    assert cargo_version == version
    assert citation_match is not None
    assert citation_match.group(1) == version
    assert f"## {version} " in changelog


def test_package_positioning_is_consistent() -> None:
    """Prevent stale or stronger claims in package-index metadata."""
    description = _toml(ROOT / "pyproject.toml")["project"]["description"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert description in readme
    assert not description.lower().startswith("validated")
