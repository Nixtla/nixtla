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
        "import nixtla._types",
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


def _intra_package_imports(relative_path):
    """The `nixtla.*` modules a source file imports, read off its AST.

    Source rather than `sys.modules`: importing *any* submodule runs
    `nixtla/__init__.py` first, which pulls in `.jobs` and `.nixtla_client`,
    so a runtime check sees the whole package loaded no matter which module
    it asked for. The layering claim is about what a file itself imports.
    """
    import ast
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    tree = ast.parse((repo_root / relative_path).read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:  # `from . import x`, `from ._http import y`
                found.add("." * node.level + (node.module or ""))
            elif (node.module or "").split(".")[0] == "nixtla":
                found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "nixtla":
                    found.add(alias.name)
    return found


def test_types_module_is_a_leaf():
    """`nixtla/_types.py` must not import from the package it sits under.

    It is the bottom of the import graph: `nixtla_client`, `jobs._namespace`
    and `_audit` all take their vocabulary from here, so anything it imported
    back would reintroduce the cycle that `nixtla/__init__.py`'s statement
    order is currently tiptoeing around.
    """
    assert _intra_package_imports("nixtla/_types.py") == set()


def test_type_aliases_are_shared_not_copied():
    """`nixtla_client` must re-export the very same objects, not redefine them.

    An `is` check on the two symbols that are real instances -- the
    `TypeAdapter` and the pydantic model -- is what distinguishes a move from a
    copy-paste. Two separate `TypeAdapter`s would both validate identically,
    and the drift would only surface the next time one of them changed.
    """
    from nixtla import _types
    from nixtla import nixtla_client

    assert nixtla_client.extra_param_checker is _types.extra_param_checker
    assert nixtla_client.FinetunedModel is _types.FinetunedModel


def test_types_module_carries_the_whole_vocabulary():
    """Every symbol the move covers has to land in `_types` and stay reachable
    from `nixtla_client`, which is where the tests and `jobs` still import
    several of them from."""
    from nixtla import _types
    from nixtla import nixtla_client

    moved = [
        "AnyDFType",
        "DistributedDFType",
        "_PositiveInt",
        "_NonNegativeInt",
        "_ExtraParamDataType",
        "extra_param_checker",
        "_Loss",
        "_Model",
        "_FinetuneDepth",
        "_Freq",
        "_FreqType",
        "_ThresholdMethod",
        "_ExplainMethod",
        "_FeatureContributionsType",
        "_MIN_SEED",
        "_MAX_SEED",
        "FinetunedModel",
        "_MAX_CONCURRENT_ASYNC_JOBS",
        "_ANOMALY_DETECTION_ENDPOINT",
        "_ONLINE_ANOMALY_DETECTION_ENDPOINT",
    ]
    missing_from_types = [n for n in moved if not hasattr(_types, n)]
    missing_from_client = [n for n in moved if not hasattr(nixtla_client, n)]
    assert missing_from_types == [], missing_from_types
    assert missing_from_client == [], missing_from_client

    # `validate_extra_params` moved too, but only `_ExtraParamDataType` ever
    # called it, so the client does not re-export it and ruff would reject the
    # unused import if it did.
    assert hasattr(_types, "validate_extra_params")
    assert not hasattr(nixtla_client, "validate_extra_params")
