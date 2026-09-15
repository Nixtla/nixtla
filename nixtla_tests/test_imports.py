import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "stmt",
    [
        "import nixtla",
        "import nixtla.jobs",
        "import nixtla.jobs._namespace",
        "from nixtla.jobs import _transport",
        "import nixtla.nixtla_client",
        "import nixtla._steps",
    ],
)
def test_imports_in_a_fresh_interpreter(stmt):
    """Each entry point has to import standalone.

    `nixtla.jobs` and `nixtla.nixtla_client` import each other, so the cycle
    resolves in only one statement order: `.jobs` before `.nixtla_client` in
    `nixtla/__init__.py`, and `from . import _transport` before the
    `..nixtla_client` imports in `nixtla/jobs/_namespace.py`. Swap either and
    the package stops importing entirely, and ruff selects only `F` here, so
    nothing reorders them today but nothing would catch it either.

    A fresh interpreter per statement is what makes this meaningful --
    in-process, the rest of the suite would already have primed `sys.modules`.
    """
    subprocess.run([sys.executable, "-c", stmt], check=True)
