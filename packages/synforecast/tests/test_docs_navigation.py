"""Checks that every Mintlify navigation item has a source document."""

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
NAVIGATION = ROOT / "docs" / "mintlify" / "docs.json"
DOCS_SOURCES = ROOT / "nbs" / "docs"
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


def _source_for(page: str, root: Path = ROOT) -> Path:
    """Map a generated Mintlify page to its checked-in source document."""
    if page == "index.html":
        return root / "README.md"
    if page.startswith("docs/"):
        relative = Path(page.removesuffix(".html")).relative_to("docs")
        notebook = root / "nbs" / "docs" / relative.with_suffix(".ipynb")
        quarto = root / "nbs" / "docs" / relative.with_suffix(".qmd")
        return notebook if notebook.is_file() else quarto
    return root / "docs" / f"{page}.md"


_MARKDOWN_LINK = re.compile(r"\]\(([^)\s]+)\)")
_ASSET_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".csv", ".parquet", ".pdf"}


def _markdown(source: Path) -> str:
    """Return the markdown prose of a notebook or Quarto document."""
    if source.suffix != ".ipynb":
        return source.read_text(encoding="utf-8")
    notebook = json.loads(source.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )


def _broken_page_links(source: Path, nbs_root: Path, root: Path) -> list[str]:
    """Relative page links in `source` that will not resolve on the published site.

    Quarto writes every page as ``<name>.html.mdx``, so its Mintlify route is
    ``<name>.html``. A bare ``[x](name)`` link resolves to ``<dir>/name``, which
    is not a page, and ``mint broken-links`` flags it. Links must carry the
    ``.html`` suffix and point at a page the build generates.
    """
    problems = []
    for target in _MARKDOWN_LINK.findall(_markdown(source)):
        page = target.split("#", 1)[0]
        if not page or page.startswith(("http://", "https://", "mailto:", "/")):
            continue
        if "_files/" in page or Path(page).suffix in _ASSET_SUFFIXES:
            continue
        if not page.endswith(".html"):
            problems.append(f"{target} (page links need the .html suffix)")
            continue
        resolved = (source.parent / page).resolve()
        try:
            route = resolved.relative_to(nbs_root).as_posix()
        except ValueError:
            problems.append(f"{target} (escapes the docs tree)")
            continue
        if not _source_for(route, root).is_file():
            problems.append(f"{target} (no source page)")
    return problems


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


def test_cross_page_links_use_published_routes() -> None:
    """Prevent bare page links that 404 on the published Mintlify site."""
    sources = sorted(DOCS_SOURCES.rglob("*.ipynb")) + sorted(
        DOCS_SOURCES.rglob("*.qmd")
    )
    assert sources, "No documentation sources found"
    broken = {
        source.relative_to(ROOT).as_posix(): problems
        for source in sources
        if (problems := _broken_page_links(source, ROOT / "nbs", ROOT))
    }
    assert not broken, "Unresolvable page links: " + "; ".join(
        f"{source}: {', '.join(problems)}" for source, problems in broken.items()
    )


def test_broken_page_links_flags_bare_and_missing_targets(tmp_path: Path) -> None:
    docs = tmp_path / "nbs" / "docs" / "capabilities"
    docs.mkdir(parents=True)
    (docs / "anomalies.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "dataset.html.md").write_text("", encoding="utf-8")
    page = docs / "changepoints.qmd"
    page.write_text(
        "See [anomalies](anomalies), [ok](./anomalies.html#top), "
        "[api](../../dataset.html), [gone](./missingness.html), "
        "[img](./changepoints_files/plot.png), [web](https://nixtla.io).",
        encoding="utf-8",
    )

    assert _broken_page_links(page, tmp_path / "nbs", tmp_path) == [
        "anomalies (page links need the .html suffix)",
        "./missingness.html (no source page)",
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
