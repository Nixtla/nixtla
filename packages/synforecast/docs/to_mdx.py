"""Convert API docs from mkdocstrings format to MDX for Mintlify."""

import re
from pathlib import Path

try:
    from mkdocstrings_parser import MkDocstringsParser
except ImportError as error:
    raise ImportError(
        "mkdocstrings-parser is required for building docs. "
        "Install the project's documentation dependency group before building."
    ) from error

DOCS_DIR = Path(__file__).parent
MINTLIFY_DIR = DOCS_DIR / "mintlify"


_CODE_SPAN = re.compile(r"<code>(.*?)</code>", re.DOTALL)


def escape_mdx_hazards(text: str) -> str:
    """Escape characters the generated MDX leaves in parser-hostile positions.

    Two patterns come out of mkdocstrings-parser and fail Mintlify's MDX parse:

    1. A union annotation such as ``str | int`` is emitted as
       ``<code>[str](#str) | [int](#int)</code>`` inside a markdown table row.
       The bare ``|`` closes the table cell, so ``<code>`` is left unclosed
       ("Expected a closing tag for `<code>`").
    2. A brace in running prose is a live MDX expression, which acorn then
       tries to parse as JavaScript ("Could not parse expression with acorn").
       This reaches the output wherever a docstring puts a brace outside
       backticks -- notably ``>>>`` example blocks, which markdown reads as a
       triple-nested blockquote rather than code.

    Both are escaped to HTML entities, which render as the original character.
    Braces inside fenced blocks and inline code spans are left alone: MDX does
    not evaluate expressions there, and escaping them would show the entity.
    """
    text = _CODE_SPAN.sub(
        lambda match: "<code>" + match.group(1).replace("|", "&#124;") + "</code>",
        text,
    )

    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        # Escape only the segments outside inline code spans.
        parts = re.split(r"(`+[^`]*`+)", line)
        out.append(
            "".join(
                part
                if part.startswith("`")
                else part.replace("{", "&#123;").replace("}", "&#125;")
                for part in parts
            )
        )
    return "\n".join(out)


def process_api_docs():
    """Process all .html.md files with ::: directives into .mdx files."""
    parser = MkDocstringsParser()

    for md_file in sorted(DOCS_DIR.glob("*.html.md")):
        print(f"Processing {md_file.name}...")
        output_file = MINTLIFY_DIR / md_file.name.replace(".html.md", ".html.mdx")
        parser.process_file(str(md_file), str(output_file))
        output_file.write_text(escape_mdx_hazards(output_file.read_text()))
        print(f"  -> {output_file.name}")


_ALERT = re.compile(r"^>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$")
_TAGLINE = re.compile(r"<h3>(.*?)</h3>")


def readme_to_index(content: str) -> str:
    """Turn the README into the docs landing page.

    The README's header is GitHub chrome: the org heading, the Slack invite,
    the centered ``<div>`` wrapper with its duplicate ``<h1>``/``<h3>``, and the
    shield badges. Mintlify already renders a title and subtitle from the
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
        if stripped.startswith(("[![", "<div", "</div>", "<h1>", "<h3>")):
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
