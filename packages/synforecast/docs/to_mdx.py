"""Convert API docs from mkdocstrings format to MDX for Mintlify."""

import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent
MINTLIFY_DIR = DOCS_DIR / "mintlify"


_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+\.)\s")


def fence_indented_blocks(text: str) -> str:
    """Turn four-space indented blocks into fenced code blocks.

    Docstrings write equations as indented blocks, which CommonMark reads as
    code. MDX has no indented code blocks, so the same text is parsed as a
    paragraph and a brace in it becomes a live expression -- a jump-diffusion
    line such as ``S_{t-} * dJ_t`` then fails acorn.

    Fencing the block instead keeps it verbatim and renders it as code, which is
    what the docstring meant. A block indented under a list item is left alone:
    there the indentation is list continuation, not code.
    """
    lines = text.split("\n")
    out: list[str] = []
    index = 0
    in_fence = False
    last_content = ""

    while index < len(lines):
        line = lines[index]
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            last_content = line.strip()
            index += 1
            continue

        indented = (line.startswith("    ") or line.startswith("\t")) and line.strip()
        starts_block = (
            not in_fence
            and indented
            and (not out or not out[-1].strip())
            and not _LIST_ITEM.match(last_content)
        )
        if not starts_block:
            if line.strip():
                last_content = line.strip()
            out.append(line)
            index += 1
            continue

        block: list[str] = []
        while index < len(lines):
            candidate = lines[index]
            if candidate.strip() and not (
                candidate.startswith("    ") or candidate.startswith("\t")
            ):
                break
            block.append(candidate)
            index += 1
        while block and not block[-1].strip():
            block.pop()
            index -= 1

        margin = min(
            len(entry) - len(entry.lstrip()) for entry in block if entry.strip()
        )
        out.append("```")
        out.extend(entry[margin:] if entry.strip() else entry for entry in block)
        out.append("```")
        last_content = "```"

    return "\n".join(out)


def escape_mdx_hazards(text: str) -> str:
    """Escape characters the generated MDX leaves in parser-hostile positions.

    Docstrings are plain text, but the generated pages are MDX, where ``{`` and
    ``<`` are syntax. Two patterns reach the output and fail Mintlify's parse:

    1. A brace in running prose is a live MDX expression, which acorn then tries
       to parse as JavaScript ("Could not parse expression with acorn").
    2. A ``<`` in running prose opens a JSX tag, so a comparison such as
       ``demand_std**2 <= demand_mean`` fails on the following character
       ("Unexpected character `=` before name"). Only a ``<`` that cannot begin
       a tag is escaped, which leaves the ``<code>``/``<details>`` markup the
       parser emits untouched.

    Both are escaped to HTML entities, which render as the original character.

    Nothing is escaped inside fenced code, inline spans, or ``$...$`` /
    ``$$...$$`` LaTeX: none of those is interpreted as MDX, and the entity
    itself would be displayed. Indented blocks are *not* exempt -- MDX, unlike
    CommonMark, has no indented code blocks, so `fence_indented_blocks` should
    run first to turn them into real fences.

    Pipes need no handling: mkdocstrings-parser escapes them as ``\\|`` before a
    table cell can be split.
    """
    # Inline code spans and LaTeX are both verbatim; capture so re.split keeps them.
    verbatim = re.compile(r"(`+[^`]*`+|\$\$[^$]*\$\$|\$[^$\n]+\$)")

    def escape_prose(segment: str) -> str:
        segment = segment.replace("{", "&#123;").replace("}", "&#125;")
        # A tag needs a name, a closing slash, or a declaration/comment bang.
        return re.sub(r"<(?![A-Za-z/!])", "&lt;", segment)

    out = []
    in_fence = False

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        out.append(
            "".join(
                part if part.startswith(("`", "$")) else escape_prose(part)
                for part in verbatim.split(line)
            )
        )
    return "\n".join(out)


def process_api_docs():
    """Process all .html.md files with ::: directives into .mdx files."""
    # Imported here so the escaping helpers stay importable (and testable)
    # without the documentation dependency group installed.
    try:
        from mkdocstrings_parser import MkDocstringsParser
    except ImportError as error:
        raise ImportError(
            "mkdocstrings-parser is required for building docs. "
            "Install the project's documentation dependency group before building."
        ) from error

    parser = MkDocstringsParser()

    for md_file in sorted(DOCS_DIR.glob("*.html.md")):
        print(f"Processing {md_file.name}...")
        output_file = MINTLIFY_DIR / md_file.name.replace(".html.md", ".html.mdx")
        parser.process_file(str(md_file), str(output_file))
        # Fence first: escaping must see indented equation blocks as real code.
        output_file.write_text(
            escape_mdx_hazards(fence_indented_blocks(output_file.read_text()))
        )
        print(f"  -> {output_file.name}")


_ALERT = re.compile(r"^>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$")
_TAGLINE = re.compile(r"<h3[^>]*>(.*?)</h3>")


def readme_to_index(content: str) -> str:
    """Turn the README into the docs landing page.

    The README's header is GitHub chrome: the org heading, the social badges,
    the Nixtla logo, and the centered ``<div>`` wrapper with its duplicate
    ``<h1>``/``<h3>``. Mintlify already renders a title and subtitle from the
    frontmatter, and the PyPI shields show "package or version not found" until
    a release exists, so all of it is dropped and the tagline is promoted to the
    frontmatter description instead.

    GitHub alert blockquotes (``> [!NOTE]``) have no Mintlify equivalent and
    would render the literal ``[!NOTE]``, so the marker line becomes a bold
    label -- the same shape Quarto emits for callouts on the other pages.
    """
    tagline_match = _TAGLINE.search(content)
    description = (
        tagline_match.group(1) if tagline_match else "Synthetic time series generation"
    )

    body = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("[![", "<div", "</div", "<h1", "<h3", "<img")):
            continue
        if stripped == "# Nixtla":
            continue
        alert = _ALERT.match(stripped)
        if alert:
            body.append(f"> **{alert.group(1).title()}**")
            body.append(">")
            continue
        body.append(line)

    # Collapse the blank lines left behind by the stripped header.
    text = "\n".join(body).lstrip("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    return (
        f'---\ntitle: "Synthetic 🧬 Forecast"\n'
        f"description: {description}\n---\n\n{text}"
    )


def process_readme():
    """Generate the index page from README.md."""
    readme_path = DOCS_DIR.parent / "README.md"
    if not readme_path.exists():
        print("WARNING: README.md not found, skipping index generation")
        return

    output_file = MINTLIFY_DIR / "index.html.mdx"
    output_file.write_text(readme_to_index(readme_path.read_text()))
    print("Created index.html.mdx from README.md")


if __name__ == "__main__":
    MINTLIFY_DIR.mkdir(parents=True, exist_ok=True)
    process_api_docs()
    process_readme()
    print("\nDone! API docs generated in docs/mintlify/")
