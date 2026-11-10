"""Check that `docs.json` and the built documentation agree, in both directions.

Mintlify resolves navigation entries against the built pages, and neither half of
a mismatch is loud on its own: a nav entry with no page is a dead link on the
site, and a built page in no nav group is published but unreachable. Both are
silent in the build log, so they are asserted here instead.

The same goes for the asset references (`logo`, `favicon`): Mintlify resolves
them against the build output, and a path pointing at nothing renders as a
missing image rather than a build failure.

Run after `make all_docs`. Exits non-zero with the offending paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINTLIFY = ROOT / "docs" / "mintlify"
CONFIG = MINTLIFY / "docs.json"


def _nav_pages(node: object) -> list[str]:
    """Collect page paths from arbitrarily nested navigation groups."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [page for item in node for page in _nav_pages(item)]
    if isinstance(node, dict):
        return [page for key in ("groups", "pages") for page in _nav_pages(node.get(key, []))]
    return []


#: Mintlify's `logo` object holds asset paths under these keys only. It also
#: accepts `href` (a URL to link the logo to) and `text`, which are not files --
#: collecting those would make this check look for `docs/mintlify/https:/...`
#: and fail Build Docs on a perfectly correct site. Since the release gate
#: requires Build Docs green, that would also block publishing.
_LOGO_ASSET_KEYS = ("light", "dark")


def _asset_refs(config: dict[str, object]) -> list[str]:
    """Collect the site-root-relative asset paths declared in ``docs.json``."""
    refs: list[str] = []
    favicon = config.get("favicon")
    if isinstance(favicon, str):
        refs.append(favicon)
    logo = config.get("logo")
    if isinstance(logo, str):
        refs.append(logo)
    elif isinstance(logo, dict):
        for key in _LOGO_ASSET_KEYS:
            value = logo.get(key)
            if isinstance(value, str):
                refs.append(value)
    return refs


def main() -> int:
    if not CONFIG.exists():
        print(f"error: {CONFIG.relative_to(ROOT)} not found", file=sys.stderr)
        return 1

    config = json.loads(CONFIG.read_text())
    declared = {f"{page}.mdx" for page in _nav_pages(config.get("navigation", {}))}
    built = {str(path.relative_to(MINTLIFY)) for path in MINTLIFY.rglob("*.mdx")}

    # The built pages are gitignored, so on a fresh clone there are none and every
    # nav entry would be reported as a dead link -- 22 spurious errors that look
    # like a broken docs.json instead of a missing build. Say what is actually
    # wrong instead.
    if not built:
        print(
            "error: no built pages found under docs/mintlify/. Run `make all_docs` "
            "first -- this check compares docs.json against the build output, which "
            "is not committed.",
            file=sys.stderr,
        )
        return 1

    missing = sorted(declared - built)
    unlisted = sorted(built - declared)
    assets = _asset_refs(config)
    absent_assets = sorted(ref for ref in assets if not (MINTLIFY / ref.lstrip("/")).is_file())

    for path in missing:
        print(f"error: docs.json lists {path}, which was not built", file=sys.stderr)
    for path in unlisted:
        print(f"error: {path} was built but is not listed in docs.json", file=sys.stderr)
    for ref in absent_assets:
        print(
            f"error: docs.json references the asset {ref}, which is not in the build "
            "output (`make brand_docs` copies the brand marks in)",
            file=sys.stderr,
        )

    if missing or unlisted or absent_assets:
        print(
            f"\n{len(missing)} dead nav entr(ies), {len(unlisted)} unreachable page(s), "
            f"{len(absent_assets)} missing asset(s)",
            file=sys.stderr,
        )
        return 1

    print(f"docs.json and the built site agree: {len(declared)} page(s), {len(assets)} asset(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
