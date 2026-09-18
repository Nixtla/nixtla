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
        "import nixtla._audit",
        "import nixtla._preprocessing",
        "import nixtla._payloads",
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


def _type_checking_imports(relative_path):
    """The `nixtla.*` modules a source file imports only under `if TYPE_CHECKING:`.

    An annotation-only import is not a runtime edge, so the layering tests
    subtract these first.
    """
    import ast
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    tree = ast.parse((repo_root / relative_path).read_text())
    found = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If)
            and getattr(node.test, "id", None) == "TYPE_CHECKING"
        ):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom):
                if inner.level:
                    found.add("." * inner.level + (inner.module or ""))
                elif (inner.module or "").split(".")[0] == "nixtla":
                    found.add(inner.module)
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


def test_audit_module_only_depends_on_http_and_types():
    """`_audit` is pure local pandas -- it must not pull in the client.

    `NixtlaClient.audit_data` and `.clean_data` are the documented entry
    points, but they only forward here; importing the client back would make
    a data-quality check depend on the HTTP stack it never touches.

    Reuses `_intra_package_imports` from Task 1 -- see its docstring for why
    this reads the source instead of `sys.modules`.
    """
    assert _intra_package_imports("nixtla/_audit.py") == {"._http", "._types"}


def test_client_audit_methods_forward_to_the_audit_module():
    """The client's methods must delegate, not carry a second implementation.

    Reading the source is blunt, but it is the only thing that catches the
    failure mode that matters here: a body left behind in `nixtla_client.py`
    alongside the new one in `_audit.py`, both passing their tests, diverging
    on the next edit.
    """
    import inspect

    from nixtla.nixtla_client import NixtlaClient

    audit_src = inspect.getsource(NixtlaClient.audit_data)
    clean_src = inspect.getsource(NixtlaClient.clean_data)

    assert "_audit.audit_data(" in audit_src
    assert "_audit.clean_data(" in clean_src
    # The implementations themselves must be gone, not merely bypassed.
    assert "_audit_duplicate_rows" not in audit_src
    assert "Fixing D001" not in clean_src


def test_audit_helpers_left_the_client_module():
    """The five checks and the severity enum live in `_audit` now, and only
    there -- `nixtla_client` must not keep aliases to them."""
    from nixtla import _audit
    from nixtla import nixtla_client

    names = [
        "AuditDataSeverity",
        "_audit_duplicate_rows",
        "_audit_missing_dates",
        "_audit_categorical_variables",
        "_audit_leading_zeros",
        "_audit_negative_values",
    ]
    assert [n for n in names if not hasattr(_audit, n)] == []
    assert [n for n in names if hasattr(nixtla_client, n)] == []


def test_the_package_ships_no_notebooks():
    """Notebooks are documentation, not library code.

    `[tool.uv.build-backend] module-root = "."` means anything under `nixtla/`
    goes into the wheel, so a stray `.ipynb` there is shipped to every user who
    pip-installs the SDK.
    """
    import pathlib

    import nixtla

    package_root = pathlib.Path(nixtla.__file__).parent
    notebooks = sorted(p.name for p in package_root.rglob("*.ipynb"))
    assert notebooks == [], f"notebooks inside the package: {notebooks}"


def test_preprocessing_module_is_a_leaf_over_http_and_types():
    """`_preprocessing` must not import the client back, or the cycle returns."""
    assert _intra_package_imports("nixtla/_preprocessing.py") == {"._http", "._types"}


def test_preprocessing_helpers_left_the_client_module():
    """The helpers live in `_preprocessing`, and `nixtla_client` keeps a name
    bound for exactly the ones it still calls -- F401 rejects an unused import
    and F821 a call to an unimported name, so the set is pinned from both sides.
    """
    from nixtla import _preprocessing
    from nixtla import nixtla_client

    moved = [
        "_date_features_by_freq",
        "_coerce_positive_int",
        "_validate_simulate_args",
        "_maybe_infer_freq",
        "_is_numeric_column",
        "_features_with_missing_values",
        "_numeric_column_array",
        "_coerce_coupled_flag",
        "_has_duplicate_keys",
        "_validate_freq_regularity",
        "_dataframe_keys_match",
        "_standardize_freq",
        "_array_tails",
        "_tail",
        "_time_col_tz",
        "_is_constant_offset_timezone",
        "_warn_non_constant_offset",
        "_times_to_iso",
        "_series_starts",
        "_partition_series",
        "_maybe_add_date_features",
        "_validate_exog",
        "_extract_categorical_exog",
        "_validate_input_size",
        "_ensure_local_dataframe",
        "_prepare_level_and_quantiles",
        "_maybe_convert_level_to_quantiles",
        "_align_future_exog_order",
        "_align_future_categorical_exog",
        "_preprocess",
        "_validate_future_exog_keys",
        "_sort_categorical_values",
        "_log_exog_features",
        "_build_exog_payload",
        "_forecast_payload_to_in_sample",
        "_get_in_sample_horizon_and_windows",
        "_maybe_add_intervals",
        "_maybe_drop_id",
        "_parse_in_sample_output",
        "_restrict_input_samples",
        "_extract_target_array",
        "_process_exog_features",
    ]
    missing = [n for n in moved if not hasattr(_preprocessing, n)]
    assert missing == [], missing

    still_on_the_client = {
        "_coerce_coupled_flag",
        "_extract_categorical_exog",
        "_forecast_payload_to_in_sample",
        "_get_in_sample_horizon_and_windows",
        "_maybe_drop_id",
        "_maybe_infer_freq",
        "_parse_in_sample_output",
        "_partition_series",
        "_preprocess",
        "_series_starts",
        "_standardize_freq",
        "_validate_freq_regularity",
        "_validate_simulate_args",
    }
    assert {n for n in moved if hasattr(nixtla_client, n)} == still_on_the_client


def test_preprocessing_helpers_are_shared_not_copied():
    """The client's names must be the very same objects, not a second copy."""
    from nixtla import _preprocessing
    from nixtla import nixtla_client

    assert nixtla_client._preprocess is _preprocessing._preprocess
    assert nixtla_client._partition_series is _preprocessing._partition_series


def test_jobs_namespace_no_longer_imports_the_client_module_at_runtime():
    """`jobs/_namespace.py` takes its preprocessing helpers from
    `.._preprocessing`, not `..nixtla_client`, so the runtime edge that closed
    the package's import cycle is gone.
    """
    path = "nixtla/jobs/_namespace.py"
    runtime = _intra_package_imports(path) - _type_checking_imports(path)
    assert "..nixtla_client" not in runtime
    assert ".._preprocessing" in runtime


def test_payloads_module_does_not_import_the_client_at_runtime():
    """`_payloads` needs the `NixtlaClient` name only for annotations, so its
    one edge back up stays under `if TYPE_CHECKING:`.
    """
    guarded = _type_checking_imports("nixtla/_payloads.py")
    assert guarded == {".nixtla_client"}
    assert _intra_package_imports("nixtla/_payloads.py") - guarded == {
        "._http",
        "._preprocessing",
        "._types",
    }


def test_prepare_methods_left_the_client_class():
    """The six builders are free functions now, with no method left behind."""
    from nixtla import _payloads
    from nixtla.nixtla_client import NixtlaClient

    builders = [
        "prepare_forecast",
        "prepare_cross_validation",
        "prepare_anomaly_detection",
        "prepare_simulate",
        "prepare_explain",
        "prepare_finetune_payload",
    ]
    missing = [n for n in builders if not hasattr(_payloads, n)]
    assert missing == [], missing

    left_behind = [n for n in builders if hasattr(NixtlaClient, "_" + n)]
    assert left_behind == [], left_behind


def test_sync_and_job_paths_share_one_builder_per_task():
    """`client.forecast()` and `client.jobs.forecast()` must assemble their
    request body with the same function, so the two paths cannot drift.
    """
    import inspect

    from nixtla.jobs._namespace import Jobs
    from nixtla.nixtla_client import NixtlaClient

    pairs = [
        ("_payloads.prepare_forecast(", NixtlaClient.forecast, Jobs.forecast),
        (
            "_payloads.prepare_cross_validation(",
            NixtlaClient.cross_validation,
            Jobs.cross_validation,
        ),
        ("_payloads.prepare_finetune_payload(", NixtlaClient.finetune, Jobs.finetune),
        (
            "_payloads.prepare_anomaly_detection(",
            NixtlaClient.detect_anomalies_online,
            Jobs.detect_anomalies,
        ),
        ("_payloads.prepare_simulate(", NixtlaClient.simulate, Jobs.simulate),
        ("_payloads.prepare_explain(", NixtlaClient.explain, Jobs.explain),
    ]
    for call, sync_method, job_method in pairs:
        assert call in inspect.getsource(sync_method), (call, sync_method.__name__)
        assert call in inspect.getsource(job_method), (call, job_method.__name__)
