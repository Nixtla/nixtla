"""Smoke tests for dataframe engines that are optional in development."""

import pytest

from synforecast.generators import RandomWalkGenerator


@pytest.mark.parametrize(
    ("engine", "module_name", "type_name"),
    [
        ("cudf", "cudf", "DataFrame"),
        ("modin", "modin.pandas", "DataFrame"),
        ("pyarrow", "pyarrow", "Table"),
    ],
)
def test_optional_dataframe_engine(
    engine: str, module_name: str, type_name: str
) -> None:
    """Generate a valid frame when an advertised optional engine is installed."""
    module = pytest.importorskip(module_name)
    expected_type = getattr(module, type_name)

    frame = RandomWalkGenerator(
        min_length=8,
        max_length=8,
        freq="D",
        engine=engine,
        seed=42,
    ).generate(n_series=2)

    assert isinstance(frame, expected_type)
    assert frame.shape[0] == 16
    if engine == "pyarrow":
        assert frame.column_names == ["unique_id", "ds", "y"]
    else:
        assert list(frame.columns) == ["unique_id", "ds", "y"]
