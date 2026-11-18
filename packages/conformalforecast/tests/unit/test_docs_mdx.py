"""Tests for the docstring -> MDX escaping helpers in ``docs/to_mdx.py``.

The helpers are imported by path rather than as a package: ``docs/`` is not
importable and must not become so. Both are importable without
``mkdocstrings-parser`` installed, so these run in the default dev environment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TO_MDX = Path(__file__).resolve().parents[2] / "docs" / "to_mdx.py"

# `docs/` is excluded from the sdist, so this module is genuinely absent when the
# suite runs from an unpacked source distribution. That is not a failure -- the
# helper under test is repo tooling, not part of the shipped package. Skipping has
# to happen at module level: the helper is loaded at import time, below.
if not _TO_MDX.exists():  # pragma: no cover - only in an unpacked sdist
    pytest.skip("docs/to_mdx.py is not shipped in the sdist", allow_module_level=True)


def _load_module():
    spec = importlib.util.spec_from_file_location("_to_mdx", _TO_MDX)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


to_mdx = _load_module()
fence_indented_blocks = to_mdx.fence_indented_blocks
escape_mdx_hazards = to_mdx.escape_mdx_hazards
readme_to_index = to_mdx.readme_to_index
parser_errors = to_mdx.parser_errors


class TestFenceIndentedBlocks:
    def test_indented_block_becomes_fenced(self):
        text = "The update rule is:\n\n    alpha_t1 = alpha_t + gamma\n\nand so on."
        assert fence_indented_blocks(text) == (
            "The update rule is:\n\n```\nalpha_t1 = alpha_t + gamma\n```\n\nand so on."
        )

    def test_common_margin_is_stripped(self):
        text = "Rule:\n\n        a = 1\n        b = 2\n"
        assert "```\na = 1\nb = 2\n```" in fence_indented_blocks(text)

    def test_list_continuation_is_left_alone(self):
        """Indentation under a list item is continuation, not code."""
        text = "- first item\n\n    continued prose\n"
        assert "```" not in fence_indented_blocks(text)

    def test_existing_fence_is_untouched(self):
        text = "```python\n    already_fenced = 1\n```\n"
        assert fence_indented_blocks(text) == text


class TestEscapeMdxHazards:
    def test_braces_in_prose_are_escaped(self):
        source = 'REQUIRED_INPUTS is frozenset({"quantile_predictions"}).'
        escaped = escape_mdx_hazards(source)
        assert "&#123;" in escaped and "&#125;" in escaped
        assert "{" not in escaped and "}" not in escaped

    def test_bare_less_than_is_escaped(self):
        assert escape_mdx_hazards("Require 0 < alpha < 1.") == ("Require 0 &lt; alpha &lt; 1.")

    @pytest.mark.parametrize("tag", ["<code>x</code>", "<br/>", "<!-- note -->"])
    def test_real_markup_survives(self, tag):
        """Only a `<` that cannot begin a tag is escaped."""
        assert escape_mdx_hazards(tag) == tag

    def test_inline_code_is_verbatim(self):
        source = "Pass `{'a': 1}` and compare `a < b`."
        assert escape_mdx_hazards(source) == source

    def test_latex_is_verbatim(self):
        source = "Coverage is $P(Y \\in C) \\geq 1 - \\alpha$ with $a < b$."
        assert escape_mdx_hazards(source) == source

    def test_fenced_code_is_verbatim(self):
        source = "```python\nd = {'a': 1}\nassert a < b\n```"
        assert escape_mdx_hazards(source) == source

    def test_composes_with_fencing(self):
        """The documented order: fence first, so equations are seen as code."""
        text = "Rule:\n\n    alpha_{t+1} = alpha_t\n"
        result = escape_mdx_hazards(fence_indented_blocks(text))
        assert "alpha_{t+1}" in result
        assert "&#123;" not in result


class TestReadmeToIndex:
    def test_frontmatter_and_fallback_description(self):
        result = readme_to_index("# conformalforecast\n\nBody text.\n")
        assert result.startswith('---\ntitle: "conformalforecast"\n')
        assert "description: Adaptive conformal prediction" in result
        assert "Body text." in result

    def test_tagline_promoted_to_description(self):
        result = readme_to_index("<h3>Calibrated intervals</h3>\n\nBody.\n")
        assert "description: Calibrated intervals" in result

    def test_github_chrome_is_stripped(self):
        readme = "# conformalforecast\n\n[![badge](x)](y)\n<img src='logo.png'>\n\nBody.\n"
        result = readme_to_index(readme)
        assert "badge" not in result and "<img" not in result

    def test_github_alert_becomes_bold_label(self):
        result = readme_to_index("> [!WARNING]\n> Unstable API.\n")
        assert "> **Warning**" in result
        assert "[!WARNING]" not in result


def test_parser_errors_are_extracted():
    text = (
        "ok\n<!-- Error generating docs for pkg.Missing: not found -->\n"
        "<!-- Error processing block: bad -->\n<!-- unrelated comment -->"
    )
    assert parser_errors(text) == [
        "Error generating docs for pkg.Missing: not found",
        "Error processing block: bad",
    ]
    assert parser_errors("<!-- unrelated comment -->") == []
