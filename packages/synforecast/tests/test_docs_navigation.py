"""Checks that every Mintlify navigation item has a source document."""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
NAVIGATION = ROOT / "docs" / "mintlify" / "docs.json"
GENERATOR_API_SOURCES = (
    ROOT / "docs" / "generators_statistical.html.md",
    ROOT / "docs" / "generators_stochastic.html.md",
    ROOT / "docs" / "generators_multivariate.html.md",
    ROOT / "docs" / "generators_domain.html.md",
    ROOT / "docs" / "generators_pretraining.html.md",
)


def _page_paths(value: Any) -> list[str]:
    """Recursively collect page paths from the Mintlify navigation tree."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [page for item in value for page in _page_paths(item)]
    if isinstance(value, dict):
        return _page_paths(value.get("pages", [])) + _page_paths(
            value.get("groups", [])
        )
    return []


def _source_for(page: str) -> Path:
    """Map a generated Mintlify page to its checked-in source document."""
    if page == "index.html":
        return ROOT / "README.md"
    if page.startswith("docs/"):
        relative = Path(page.removesuffix(".html")).relative_to("docs")
        notebook = ROOT / "nbs" / "docs" / relative.with_suffix(".ipynb")
        quarto = ROOT / "nbs" / "docs" / relative.with_suffix(".qmd")
        return notebook if notebook.is_file() else quarto
    return ROOT / "docs" / f"{page}.md"


def test_mintlify_navigation_pages_have_sources() -> None:
    """Prevent navigation links to pages the docs build cannot generate."""
    configuration = json.loads(NAVIGATION.read_text(encoding="utf-8"))
    pages = _page_paths(configuration["navigation"])
    missing = {
        page: _source_for(page) for page in pages if not _source_for(page).is_file()
    }
    assert not missing, "Missing documentation sources: " + ", ".join(
        f"{page} -> {source.relative_to(ROOT)}" for page, source in missing.items()
    )


def test_page_paths_collects_pages_and_groups() -> None:
    navigation = {
        "pages": ["overview"],
        "groups": [{"pages": ["docs/getting-started/quickstart.html"]}],
    }

    assert _page_paths(navigation) == [
        "overview",
        "docs/getting-started/quickstart.html",
    ]


def test_every_exported_generator_has_api_reference() -> None:
    """Keep the generated API reference aligned with the public generator API."""
    import synforecast.generators as generators

    api_sources = "\n".join(
        source.read_text(encoding="utf-8") for source in GENERATOR_API_SOURCES
    )
    missing = []
    for name in generators.__all__:
        generator = getattr(generators, name)
        directive = f"::: {generator.__module__}.{generator.__name__}"
        if directive not in api_sources:
            missing.append(name)

    assert not missing, "Missing generator API references: " + ", ".join(missing)
