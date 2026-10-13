"""Execute the README's quick-start snippet.

The README carries a hand-maintained near-duplicate of the quick-start
notebook, and nothing checked that it runs. It is the first code a new user
copies, so a stale API in it is the most expensive kind of documentation bug.

This extracts the first ``python`` fenced block after the "## Quick start"
heading and executes it in a fresh namespace, then asserts the shapes the
snippet's trailing lines advertise.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("statsforecast")

README = Path(__file__).resolve().parents[2] / "README.md"

# `docs/` and the repo root are excluded from the sdist, so the README is absent
# when the suite runs from an unpacked source distribution.
if not README.exists():  # pragma: no cover - only in an unpacked sdist
    pytest.skip("README.md is not shipped in the sdist", allow_module_level=True)


def _quickstart_snippet() -> str:
    """Return the first python code block under the '## Quick start' heading."""
    text = README.read_text()
    start = text.index("## Quick start")
    end = text.find("\n## ", start + 1)
    section = text[start : end if end != -1 else len(text)]
    blocks = re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL)
    assert blocks, "no python code block found under '## Quick start' in README.md"
    return blocks[0]


def test_readme_quickstart_runs() -> None:
    namespace: dict[str, Any] = {}
    exec(compile(_quickstart_snippet(), "README.md::quickstart", "exec"), namespace)  # noqa: S102

    prediction = namespace["prediction"]
    n_series = namespace["n_series"]
    horizon = namespace["horizon"]

    assert prediction.point.shape == (n_series, 1, horizon)
    assert prediction.interval.shape == (n_series, 1, horizon, 2)
    assert np.isfinite(prediction.point).all()
    assert np.isfinite(prediction.interval).all()
    assert (prediction.interval[..., 0] <= prediction.point).all()
    assert (prediction.point <= prediction.interval[..., 1]).all()
