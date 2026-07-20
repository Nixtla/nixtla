"""Convert API docs from mkdocstrings format to MDX for Mintlify."""

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


def process_api_docs():
    """Process all .html.md files with ::: directives into .mdx files."""
    parser = MkDocstringsParser()

    for md_file in sorted(DOCS_DIR.glob("*.html.md")):
        print(f"Processing {md_file.name}...")
        output_file = MINTLIFY_DIR / md_file.name.replace(".html.md", ".html.mdx")
        parser.process_file(str(md_file), str(output_file))
        print(f"  -> {output_file.name}")


def process_readme():
    """Copy README.md as the index page, skipping badge lines."""
    readme_path = DOCS_DIR.parent / "README.md"
    if not readme_path.exists():
        print("WARNING: README.md not found, skipping index generation")
        return

    content = readme_path.read_text()
    lines = content.split("\n")

    # Skip badge/shield lines at the top
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith(
            ("[![", "[!", "<a", "<p", "<div", "---")
        ):
            start_idx = i
            break

    cleaned = "\n".join(lines[start_idx:])

    # Add frontmatter
    output = f"---\ntitle: SynForecast\ndescription: Synthetic Time Series Generation\n---\n\n{cleaned}"

    output_file = MINTLIFY_DIR / "index.html.mdx"
    output_file.write_text(output)
    print("Created index.html.mdx from README.md")


if __name__ == "__main__":
    MINTLIFY_DIR.mkdir(parents=True, exist_ok=True)
    process_api_docs()
    process_readme()
    print("\nDone! API docs generated in docs/mintlify/")
