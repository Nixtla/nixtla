"""Checks for the MDX post-processing that builds the API reference.

The generated pages are MDX, so `{` and `<` in a docstring are syntax rather
than text. `escape_mdx_hazards` neutralizes them without touching code, where
MDX interprets nothing and an entity would be displayed literally.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("to_mdx", ROOT / "docs" / "to_mdx.py")
assert _spec is not None and _spec.loader is not None
to_mdx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(to_mdx)

escape_mdx_hazards = to_mdx.escape_mdx_hazards
readme_to_index = to_mdx.readme_to_index
fence_indented_blocks = to_mdx.fence_indented_blocks


class TestEscapesProse:
    def test_brace_becomes_an_entity(self) -> None:
        assert escape_mdx_hazards("ids follow {id}_aug_{i}") == (
            "ids follow &#123;id&#125;_aug_&#123;i&#125;"
        )

    def test_comparison_operator_is_escaped(self) -> None:
        """A bare `<` opens a JSX tag, so `<=` fails on the `=`."""
        assert escape_mdx_hazards("Poisson when std**2 <= mean") == (
            "Poisson when std**2 &lt;= mean"
        )

    def test_lone_less_than_is_escaped(self) -> None:
        assert escape_mdx_hazards("alpha < 2 gives heavy tails") == (
            "alpha &lt; 2 gives heavy tails"
        )


class TestLeavesMarkupAlone:
    @pytest.mark.parametrize(
        "line",
        [
            "`freq` | <code>[str](#str)</code> | Frequency | *required*",
            '<details class="example" open markdown="1">',
            "</details>",
            "<!-- a comment -->",
        ],
    )
    def test_html_the_parser_emits_survives(self, line: str) -> None:
        assert escape_mdx_hazards(line) == line

    def test_escaped_pipe_is_not_touched(self) -> None:
        """mkdocstrings-parser already escapes pipes; re-escaping yields `\\&#124;`."""
        line = r"`freq` | <code>[str](#str) \| [int](#int)</code> | Frequency"
        assert escape_mdx_hazards(line) == line


class TestLeavesCodeAlone:
    def test_inline_code_span(self) -> None:
        line = "the recursion `x_{t-1} < x_t` holds"
        assert escape_mdx_hazards(line) == line

    def test_fenced_block(self) -> None:
        text = (
            "prose {here}\n```python\nd = {'a': 1}\nif a < b:\n    pass\n```\nafter {x}"
        )
        result = escape_mdx_hazards(text)
        assert "d = {'a': 1}" in result
        assert "if a < b:" in result
        assert "prose &#123;here&#125;" in result
        assert "after &#123;x&#125;" in result

    def test_indented_block_is_escaped_unless_fenced_first(self) -> None:
        """MDX has no indented code blocks, so indentation alone is not exempt."""
        text = "variance\n\n    s2_t = w + a * e_{t-i}^2\n"
        assert "&#123;t-i&#125;" in escape_mdx_hazards(text)

    def test_indentation_inside_a_paragraph_is_not_code(self) -> None:
        text = "a wrapped sentence\n    continuing with {braces}"
        assert "&#123;braces&#125;" in escape_mdx_hazards(text)


class TestIdempotent:
    def test_running_twice_changes_nothing_further(self) -> None:
        text = "std**2 <= mean and {id}\n\n    literal = {kept}\n"
        once = escape_mdx_hazards(text)
        assert escape_mdx_hazards(once) == once


class TestReadmeToIndex:
    def test_github_chrome_is_dropped_and_tagline_promoted(self) -> None:
        readme = (
            "# Nixtla\n\n"
            "[![Slack](https://img.shields.io/badge/Slack-x)](https://example.com)\n\n"
            '<div align="center">\n'
            "<h1>SynForecast</h1>\n"
            "<h3>Fast synthetic time series</h3>\n\n"
            "[![CI](https://example.com/badge.svg)](https://example.com)\n\n"
            "**SynForecast** generates panels.\n"
            "</div>\n\n"
            "> [!NOTE]\n"
            "> Alpha software.\n\n"
            "## Installation\n"
        )
        out = readme_to_index(readme)
        assert "description: Fast synthetic time series" in out
        for dropped in ("# Nixtla", "![Slack]", "<div", "<h1>", "<h3>", "[!NOTE]"):
            assert dropped not in out
        assert "> **Note**" in out
        assert "**SynForecast** generates panels." in out
        assert "## Installation" in out


class TestLeavesMathAlone:
    """The Nixtlaverse writes equations as LaTeX, so `$...$` must stay verbatim."""

    def test_inline_math_keeps_its_braces(self) -> None:
        line = r"the recursion $x_{t} = \phi x_{t-1}$ holds"
        assert escape_mdx_hazards(line) == line

    def test_display_math_keeps_its_braces(self) -> None:
        line = r"$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2$$"
        assert escape_mdx_hazards(line) == line

    def test_prose_around_math_is_still_escaped(self) -> None:
        result = escape_mdx_hazards(r"when {x} then $y_{t}$ and {z}")
        assert result == r"when &#123;x&#125; then $y_{t}$ and &#123;z&#125;"


class TestFenceIndentedBlocks:
    """MDX has no indented code blocks, so they must become real fences."""

    def test_equation_block_is_fenced_and_dedented(self) -> None:
        text = (
            "conditional variance\n"
            "\n"
            "    sigma2_t = omega + sum_i alpha_i * eps_{t-i}^2\n"
            "                     + sum_j beta_j * sigma2_{t-j}\n"
            "\n"
            "Stationarity requires ...\n"
        )
        result = fence_indented_blocks(text)
        assert "```\nsigma2_t = omega + sum_i alpha_i * eps_{t-i}^2\n" in result
        # relative alignment of the continuation line is preserved
        assert "\n                 + sum_j beta_j * sigma2_{t-j}\n```" in result
        assert "Stationarity requires ..." in result

    def test_fencing_then_escaping_leaves_the_equation_verbatim(self) -> None:
        text = "the model is:\n\n    dS_t = mu * S_t * dt + S_{t-} * dJ_t\n"
        result = escape_mdx_hazards(fence_indented_blocks(text))
        assert "dS_t = mu * S_t * dt + S_{t-} * dJ_t" in result
        assert "&#123;" not in result

    def test_list_continuation_is_not_fenced(self) -> None:
        """Indentation under a list item is continuation, not code."""
        text = "- first item\n\n    continuation of the item\n"
        assert "```" not in fence_indented_blocks(text)

    def test_content_inside_an_existing_fence_is_untouched(self) -> None:
        text = "```python\n    already_indented = 1\n```\n"
        assert fence_indented_blocks(text) == text

    def test_idempotent(self) -> None:
        text = "model:\n\n    x = {y}\n\nafter\n"
        once = fence_indented_blocks(text)
        assert fence_indented_blocks(once) == once
