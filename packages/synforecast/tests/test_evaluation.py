"""Coverage arithmetic, reference stability, retention, and panel contracts."""

import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest

from synforecast._coverage import grid_occupancy, occupancy_scores
from synforecast._features import FEATURE_NAMES, compute_feature_set
from synforecast._features import compute_targeting_features as targeting_features
from synforecast.evaluation import (
    compare_feature_coverage,
    compute_features,
    feature_coverage,
)


def frame(engine, data):
    return pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)


def feature_frame(engine, x=(-2.0, 0.0, 2.0), **extra):
    return frame(
        engine, {"unique_id": [f"id{i}" for i in range(len(x))], "x": list(x), **extra}
    )


def panel(engine):
    rng = np.random.default_rng(12)
    return frame(
        engine,
        {
            "unique_id": np.repeat(["b", "a", "c"], 48),
            "ds": np.tile(np.arange(48)[::-1], 3),
            "y": rng.normal(size=144),
        },
    )


def test_grid_denominators_and_boundaries():
    edges = (np.linspace(0, 3, 4), np.linspace(0, 3, 4))
    real, cells, inside = grid_occupancy(
        np.array([[0, 0], [3, 3], [1, 2], [4, 1], [-1, 0]]), edges
    )
    np.testing.assert_array_equal(inside, [True, True, True, False, False])
    np.testing.assert_array_equal(cells[:3], [[0, 0], [2, 2], [1, 2]])
    syn, _, _ = grid_occupancy(np.array([[0.1, 0.1], [1.2, 0.2]]), edges)
    assert occupancy_scores(real, syn) == pytest.approx((2 / 9, 1 / 9, 2 / 3))
    assert occupancy_scores(real, np.zeros_like(real)) == pytest.approx((3 / 9, 0, 1))
    assert occupancy_scores(real, real) == (0, 0, 0)


def test_identity_rank_one_and_id_mapping(engine):
    real = feature_frame(engine)
    result = feature_coverage(real, real, precomputed=True)
    assert result.miscoverage == result.reverse_miscoverage == 0
    assert result.uncovered_real_series_fraction == 0
    assert result.real_diagnostics.retained_ids == ("id0", "id1", "id2")
    assert result.explained_variance == (1, 0)
    assert result.n_synthetic_out_of_range == 0
    np.testing.assert_array_equal(result.real_embedding[:, 1], 0)
    assert np.all(np.diff(result.grid_edges[1]) > 0)


def test_reference_space_stability_and_constant_deviations(engine):
    real = feature_frame(engine, constant=[1.0] * 3)
    synthetic = feature_frame(engine, (-1.0, 0.0, 1.0), constant=[1.0, 2.0, 1.0])
    first = compare_feature_coverage(real, {"first": synthetic}, precomputed=True)[
        "first"
    ]
    later = compare_feature_coverage(
        real,
        {"first": synthetic, "extreme": feature_frame(engine, (1e6,), constant=[50.0])},
        precomputed=True,
    )["first"]
    assert first.space_id == later.space_id
    assert first.metadata["reference_id"] == later.metadata["reference_id"]
    np.testing.assert_array_equal(first.real_embedding, later.real_embedding)
    np.testing.assert_array_equal(first.synthetic_embedding, later.synthetic_embedding)
    np.testing.assert_array_equal(first.real_occupancy, later.real_occupancy)
    assert first.miscoverage == later.miscoverage
    assert first.features_used == ("x",)
    assert first.constant_features_dropped == ("constant",)
    assert first.synthetic_constant_feature_deviations == {"constant": 1}


def test_out_of_range_and_pooled_grid(engine):
    real = feature_frame(engine)
    matching = feature_coverage(real, real, precomputed=True, n_bins=4)
    extra = feature_frame(engine, (-2.0, 0.0, 2.0, 1000.0))
    result = feature_coverage(real, extra, precomputed=True, n_bins=4)
    assert result.space_id == matching.space_id
    assert result.miscoverage == result.reverse_miscoverage == 0
    assert result.n_synthetic_out_of_range == 1
    assert result.synthetic_out_of_range_fraction == 0.25
    pooled = feature_coverage(
        real, extra, precomputed=True, grid_range="pooled", n_bins=4
    )
    assert pooled.space_id != result.space_id
    assert pooled.grid_edges[0][-1] > result.grid_edges[0][-1]
    assert pooled.n_synthetic_out_of_range == 0
    all_out = feature_coverage(
        real, feature_frame(engine, (100.0, 200.0)), precomputed=True
    )
    assert (
        all_out.uncovered_real_cell_fraction
        == all_out.uncovered_real_series_fraction
        == 1
    )
    assert all_out.reverse_miscoverage == 0
    assert all_out.synthetic_out_of_range_fraction == 1


def test_real_series_fraction_weights_series_not_cells(engine):
    real = feature_frame(engine, (-1.0, -1.0, -1.0, 1.0))
    synthetic = feature_frame(engine, (1.0,))
    result = feature_coverage(real, synthetic, precomputed=True, n_bins=2)
    assert result.uncovered_real_cell_fraction == 0.5
    assert result.uncovered_real_series_fraction == 0.75
    assert result.uncovered_real_ids == ("id0", "id1", "id2")


def test_precomputed_column_alignment_and_selection(engine):
    real = feature_frame(engine, y=[0.0, 2.0, 1.0])
    swapped = frame(
        engine,
        {
            "y": [0.0, 2.0, 1.0],
            "unique_id": ["id0", "id1", "id2"],
            "x": [-2.0, 0.0, 2.0],
        },
    )
    assert feature_coverage(real, swapped, precomputed=True).miscoverage == 0
    with pytest.raises(ValueError, match="columns must match"):
        feature_coverage(real, feature_frame(engine), precomputed=True)
    assert (
        feature_coverage(
            real, feature_frame(engine), precomputed=True, features=["x"]
        ).miscoverage
        == 0
    )


def test_missing_diagnostics_and_reference_retention(engine, caplog):
    real = feature_frame(engine, (-2.0, np.nan, 2.0))
    synthetic = feature_frame(engine, (-2.0, np.inf, 2.0))
    with pytest.raises(ValueError, match="non-finite"):
        feature_coverage(real, synthetic, precomputed=True)
    with caplog.at_level(logging.WARNING, logger="synforecast.evaluation"):
        result = feature_coverage(real, synthetic, precomputed=True, missing="drop")
    assert len(caplog.records) == 2
    assert result.real_diagnostics.n_input == 3
    assert result.real_diagnostics.retained_ids == ("id0", "id2")
    assert result.real_diagnostics.dropped_ids == ("id1",)
    assert result.synthetic_diagnostics.drop_reasons == {"id1": ("x",)}
    assert result.miscoverage == 0
    with pytest.raises(ValueError, match="no retained"):
        feature_coverage(
            real, feature_frame(engine, (np.nan,)), precomputed=True, missing="drop"
        )


@pytest.mark.parametrize("x", [(1.0,), (1.0, 1.0)])
def test_fitting_population_requirements(engine, x):
    with pytest.raises(ValueError, match="at least two|No variable"):
        feature_coverage(
            feature_frame(engine, x), feature_frame(engine), precomputed=True
        )
    result = feature_coverage(
        feature_frame(engine, x), feature_frame(engine), precomputed=True, fit="pooled"
    )
    assert result.fit == "pooled"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_bins": 0},
        {"n_bins": True},
        {"features": []},
        {"features": "x"},
        {"features": ["x", "x"]},
        {"features": ["missing"]},
        {"fit": "invalid"},
        {"embedding": "invalid"},
        {"grid_range": "invalid"},
        {"missing": "impute"},
        {"n_jobs": 0},
        {"n_jobs": -2},
        {"n_jobs": None},
        {"n_jobs": True},
        {"embedding": "tsne"},
        {"embedding": "tsne", "fit": "pooled"},
    ],
)
def test_invalid_options(kwargs):
    real = feature_frame("pandas")
    with pytest.raises(ValueError):
        feature_coverage(real, real, precomputed=True, **kwargs)


def test_precomputed_invalid_ids_and_dtypes(engine):
    real = feature_frame(engine)
    for bad in (
        frame(engine, {"unique_id": ["x", "x"], "x": [1.0, 2.0]}),
        frame(engine, {"unique_id": ["x", None], "x": [1.0, 2.0]}),
        frame(engine, {"unique_id": ["x"], "x": ["text"]}),
    ):
        with pytest.raises(ValueError):
            feature_coverage(real, bad, precomputed=True)


def test_native_panel_precomputed_and_workers(engine):
    real = panel(engine)
    extracted = compute_features(real, seasonal_period=4)
    assert type(extracted) is type(real)
    assert list(extracted.columns) == ["unique_id", *FEATURE_NAMES]
    expected = feature_coverage(extracted, extracted, precomputed=True)
    actual = feature_coverage(real, real, seasonal_period=4, n_jobs=2)
    np.testing.assert_allclose(actual.real_embedding, expected.real_embedding)
    assert actual.miscoverage == 0
    assert actual.metadata["parameters"]["schema"] == "native_v1"
    assert actual.metadata["parameters"]["effective_window_size"] == 4
    native = (
        extracted.to_dict(as_series=False)
        if engine == "polars"
        else extracted.to_dict("list")
    )
    # Independent call to the existing MAR helper verifies its four-value contract.
    if engine == "polars":
        values = real.filter(pl.col("unique_id") == "a").sort("ds")["y"].to_numpy()
    else:
        values = real[real.unique_id == "a"].sort_values("ds").y.to_numpy()
    for name, value in targeting_features(values, 4).items():
        assert native[name][0] == pytest.approx(value)


def test_custom_columns_and_short_series(engine, caplog):
    df = frame(
        engine, {"series": ["b", "a", "a"], "time": [0, 1, 0], "value": [0.0, 1.0, 2.0]}
    )
    with caplog.at_level(logging.WARNING, logger="synforecast.evaluation"):
        result = compute_features(
            df, id_col="series", time_col="time", target_col="value", features=["acf1"]
        )
    assert len(result) == 2
    assert list(result.columns) == ["series", "acf1"]
    assert len(caplog.records) == 1
    assert "2 series" in caplog.records[0].message


@pytest.mark.parametrize(
    "data",
    [
        {"unique_id": ["a", "a"], "ds": [1, 1], "y": [1.0, 2.0]},
        {"unique_id": [None], "ds": [1], "y": [1.0]},
        {"unique_id": ["a"], "ds": [None], "y": [1.0]},
        {"unique_id": ["a"], "ds": [1], "y": [np.inf]},
        {"unique_id": ["a"], "ds": [1], "y": ["invalid"]},
    ],
)
def test_invalid_raw_panels(engine, data):
    with pytest.raises(ValueError):
        compute_features(frame(engine, data))


def test_pandas_polars_agreement():
    pandas = feature_coverage(panel("pandas"), panel("pandas"), seasonal_period=4)
    polars = feature_coverage(panel("polars"), panel("polars"), seasonal_period=4)
    np.testing.assert_array_equal(pandas.real_embedding, polars.real_embedding)
    assert pandas.space_id == polars.space_id


@pytest.mark.parametrize("column", ["unique_id", "ds"])
def test_nonfinite_numeric_panel_identifiers(engine, column):
    data = {"unique_id": [0.0, 0.0, 0.0], "ds": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]}
    data[column][0] = np.nan
    with pytest.raises(ValueError):
        compute_features(frame(engine, data))


def test_nonfinite_numeric_feature_identifiers(engine):
    real = feature_frame(engine)
    invalid = frame(engine, {"unique_id": [np.nan], "x": [1.0]})
    with pytest.raises(ValueError):
        feature_coverage(real, invalid, precomputed=True)


def test_tsne_seed_and_small_population():
    pytest.importorskip("sklearn")
    real = feature_frame("pandas")
    with pytest.raises(ValueError, match="more than 30"):
        feature_coverage(
            real, real, precomputed=True, embedding="tsne", fit="pooled", seed=4
        )
    rng = np.random.default_rng(4)
    real = feature_frame("pandas", rng.normal(size=20), y=rng.normal(size=20).tolist())
    kwargs = {"precomputed": True, "embedding": "tsne", "fit": "pooled", "seed": 4}
    first = feature_coverage(real, real, **kwargs)
    second = feature_coverage(real, real, **kwargs)
    np.testing.assert_array_equal(first.real_embedding, second.real_embedding)
    assert first.space_id == second.space_id


def test_plot_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = feature_frame("pandas")
    result = feature_coverage(real, feature_frame("pandas", (100.0,)), precomputed=True)
    ax = result.plot()
    assert "outside grid: 1" in ax.get_title()
    plt.close(ax.figure)


def test_window_configuration_is_validated_recorded_and_forwarded(engine):
    real = panel(engine)
    result = feature_coverage(real, real, window_size=5)
    assert result.metadata["parameters"]["window_size"] == 5
    assert result.miscoverage == 0
    default = feature_coverage(real, real)
    assert result.space_id != default.space_id
    raw = compute_features(real, window_size=5)
    cached = feature_coverage(raw, raw, precomputed=True)
    np.testing.assert_allclose(result.real_embedding, cached.real_embedding)
    for invalid in (0, 1, True, 2.5):
        with pytest.raises(ValueError, match="window_size"):
            compute_features(real, window_size=invalid)


def test_drop_reasons_name_only_the_offending_features(engine):
    """Attribution must select per feature, not list the whole schema."""
    real = frame(
        engine,
        {
            "unique_id": ["id0", "id1", "id2", "id3"],
            "a": [-2.0, np.nan, 2.0, 0.5],
            "b": [-2.0, 1.0, 2.0, np.nan],
            "c": [-2.0, np.nan, 2.0, 0.5],
        },
    )
    result = feature_coverage(real, real, precomputed=True, missing="drop")
    assert result.real_diagnostics.drop_reasons == {
        "id1": ("a", "c"),
        "id3": ("b",),
    }
    assert result.real_diagnostics.retained_ids == ("id0", "id2")


def test_uncovered_ids_align_with_retained_ids_after_drops(engine):
    """Dropping must not shift the id/occupancy pairing for uncovered series."""
    real = frame(
        engine,
        {
            "unique_id": ["keep_lo", "drop", "keep_hi", "keep_mid"],
            "a": [-5.0, np.nan, 5.0, 0.0],
            "b": [-5.0, 1.0, 5.0, 0.0],
        },
    )
    synthetic = frame(
        engine,
        {
            "unique_id": ["s0", "s1"],
            "a": [-5.0, 5.0],
            "b": [-5.0, 5.0],
        },
    )
    result = feature_coverage(real, synthetic, precomputed=True, missing="drop")
    assert result.real_diagnostics.retained_ids == ("keep_lo", "keep_hi", "keep_mid")
    # The middle series is the only real point no synthetic corner covers.
    assert result.uncovered_real_ids == ("keep_mid",)


def test_pooled_fit_uses_the_combined_population(engine):
    """fit='pooled' must standardize on both corpora, not on real alone."""
    real = feature_frame(engine, (-1.0, 0.0, 1.0))
    synthetic = feature_frame(engine, (99.0, 100.0, 101.0))
    on_real = feature_coverage(real, synthetic, precomputed=True, fit="real")
    pooled = feature_coverage(real, synthetic, precomputed=True, fit="pooled")
    real_mean = on_real.metadata["parameters"]["mean"][0]
    pooled_mean = pooled.metadata["parameters"]["mean"][0]
    assert real_mean == pytest.approx(0.0)
    assert pooled_mean == pytest.approx(50.0)
    assert on_real.space_id != pooled.space_id


def test_pooled_fit_with_real_grid_range(engine):
    """fit and grid_range are independent knobs; this pairing is valid."""
    real = feature_frame(engine, (-1.0, 0.0, 1.0))
    synthetic = feature_frame(engine, (5.0, 6.0, 7.0))
    result = feature_coverage(
        real, synthetic, precomputed=True, fit="pooled", grid_range="real"
    )
    assert result.fit == "pooled" and result.grid_range == "real"
    # The synthetic corpus sits entirely outside the real-only grid.
    assert result.n_synthetic_out_of_range == 3
    assert result.synthetic_out_of_range_fraction == pytest.approx(1.0)


def test_precomputed_rejects_native_extraction_keywords(engine):
    real = feature_frame(engine)
    for kwargs in (
        {"seasonal_period": 12},
        {"freq": "MS"},
        {"window_size": 5},
        {"time_col": "time"},
        {"target_col": "value"},
    ):
        with pytest.raises(ValueError, match="cannot be combined with precomputed"):
            feature_coverage(real, real, precomputed=True, **kwargs)


@pytest.mark.parametrize("bad", [0, -4, 12.0, True])
def test_seasonal_period_is_validated(bad):
    with pytest.raises(ValueError, match="seasonal_period must be an integer >= 1"):
        compute_features(panel("pandas"), seasonal_period=bad)


@pytest.mark.parametrize(
    "synthetics",
    [{}, [1], {1: None}],
)
def test_compare_rejects_invalid_synthetics_mapping(synthetics):
    with pytest.raises(ValueError, match="non-empty mapping"):
        compare_feature_coverage(feature_frame("pandas"), synthetics, precomputed=True)


def test_tsne_splits_unequal_corpora_at_the_right_boundary():
    """An equal-size real/real split would hide an off-by-one or swap."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(7)
    real = feature_frame("pandas", rng.normal(size=35), y=rng.normal(size=35).tolist())
    values = rng.normal(size=20)
    values[3] = np.nan
    synthetic = feature_frame("pandas", values, y=rng.normal(size=20).tolist())
    result = feature_coverage(
        real,
        synthetic,
        precomputed=True,
        embedding="tsne",
        fit="pooled",
        seed=11,
        missing="drop",
    )
    assert result.real_embedding.shape[0] == 35
    # One synthetic row is dropped before embedding, so the split must be 35/19.
    assert result.synthetic_embedding.shape[0] == 19
    assert len(result.synthetic_diagnostics.retained_ids) == 19


def test_reference_id_tracks_the_real_feature_values(engine):
    """Stability alone would be satisfied by a constant; pin sensitivity too."""
    base = feature_coverage(
        feature_frame(engine), feature_frame(engine), precomputed=True
    ).metadata["reference_id"]
    same = feature_coverage(
        feature_frame(engine), feature_frame(engine, (-3.0, 1.0, 4.0)), precomputed=True
    ).metadata["reference_id"]
    # A different synthetic corpus must not move the reference fingerprint.
    assert same == base
    for changed in [(-2.0, 0.0, 2.5), (-2.0, 0.0, 2.0, 3.0)]:
        altered = feature_coverage(
            feature_frame(engine, changed), feature_frame(engine), precomputed=True
        ).metadata["reference_id"]
        assert altered != base


def unequal_panel(engine, ds="int"):
    """Three series of different lengths, rows shuffled, times descending."""
    rng = np.random.default_rng(21)
    lengths = {"b": 30, "a": 55, "c": 41}
    ids = np.concatenate([[uid] * n for uid, n in lengths.items()])
    steps = np.concatenate([np.arange(n)[::-1] for n in lengths.values()])
    times = (
        steps
        if ds == "int"
        else pd.Timestamp("2000-01-01")
        + pd.to_timedelta(steps, unit="h" if ds == "hourly" else "D") * 30
    )
    order = rng.permutation(len(ids))
    data = {
        "unique_id": ids[order],
        "ds": np.asarray(times)[order],
        "y": rng.normal(size=len(ids))[order],
    }
    return frame(engine, data)


def test_unequal_lengths_map_to_their_own_series(engine):
    df = unequal_panel(engine)
    features = compute_features(df, seasonal_period=4)
    rows = features.to_dict("records") if engine == "pandas" else features.to_dicts()
    assert [row["unique_id"] for row in rows] == ["a", "b", "c"]
    source = df if engine == "pandas" else df.to_pandas()
    for row in rows:
        values = source[source.unique_id == row["unique_id"]].sort_values("ds").y
        expected = compute_feature_set(values.to_numpy(), 4)
        for name, value in expected.items():
            assert row[name] == pytest.approx(value, nan_ok=True), name


def test_pandas_output_has_a_fresh_range_index():
    features = compute_features(panel("pandas"), seasonal_period=4)
    pd.testing.assert_index_equal(features.index, pd.RangeIndex(3))


def test_datetime_times_of_any_frequency_match_integer_times(engine):
    expected = compute_features(unequal_panel(engine), seasonal_period=4)
    for ds in ("hourly", "daily"):
        actual = compute_features(unequal_panel(engine, ds), seasonal_period=4)
        left = actual.to_pandas() if engine == "polars" else actual
        right = expected.to_pandas() if engine == "polars" else expected
        pd.testing.assert_frame_equal(left, right)


def test_missing_datetime_raises():
    df = unequal_panel("pandas", "daily")
    df.loc[df.index[0], "ds"] = pd.NaT
    with pytest.raises(ValueError, match="null"):
        compute_features(df)


def test_freq_derives_the_period_like_the_presets(engine):
    df = unequal_panel(engine)
    by_freq = compute_features(df, freq="QS")
    by_period = compute_features(df, seasonal_period=4)
    left = by_freq.to_pandas() if engine == "polars" else by_freq
    right = by_period.to_pandas() if engine == "polars" else by_period
    pd.testing.assert_frame_equal(left, right)
    # An explicit period wins over freq, as in the presets.
    explicit = compute_features(df, freq="MS", seasonal_period=4)
    explicit = explicit.to_pandas() if engine == "polars" else explicit
    pd.testing.assert_frame_equal(explicit, right)


@pytest.mark.parametrize("kwargs", [{"seasonal_period": 1}, {"freq": "YS"}])
def test_period_one_means_no_seasonality(kwargs):
    df = unequal_panel("pandas")
    pd.testing.assert_frame_equal(compute_features(df, **kwargs), compute_features(df))


def test_integer_freq_is_rejected():
    with pytest.raises(ValueError, match="offset alias"):
        compute_features(panel("pandas"), freq=1)


def test_undeclared_seasonality_is_logged_when_it_drops_out(caplog):
    df = unequal_panel("pandas")
    with caplog.at_level(logging.WARNING, logger="synforecast.evaluation"):
        result = feature_coverage(df, df)
    assert {"seasonal_strength", "seas_acf1"} <= set(result.constant_features_dropped)
    assert any("No seasonal period declared" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="synforecast.evaluation"):
        result = feature_coverage(df, df, freq="QS")
    assert "seasonal_strength" in result.features_used
    assert not any("No seasonal period" in r.message for r in caplog.records)
    assert result.metadata["parameters"]["seasonal_period"] == 4


def test_results_compare_by_identity_and_own_their_metadata(engine):
    real = feature_frame(engine)
    results = compare_feature_coverage(real, {"a": real, "b": real}, precomputed=True)
    first, second = results["a"], results["b"]
    assert first != second
    assert first == first
    assert len({first, second, first.real_diagnostics}) == 3
    first.metadata["parameters"]["features"] = ("changed",)
    assert second.metadata["parameters"]["features"] == ("x",)


def test_summary_tabulates_scalar_scores(engine):
    real = feature_frame(engine)
    synthetic = feature_frame(engine, (-2.0, 2.0, 50.0))
    result = feature_coverage(real, synthetic, precomputed=True)
    summary = result.summary()
    assert summary["miscoverage"] == result.miscoverage
    assert summary["n_synthetic_retained"] == 3
    assert summary["n_real_input"] == 3
    assert summary["explained_variance"] == pytest.approx(1.0)
    table = pd.DataFrame({"synthetic": summary}).T
    assert table.loc["synthetic", "synthetic_out_of_range_fraction"] == pytest.approx(
        1 / 3
    )


def test_native_missing_drop_end_to_end(engine):
    rng = np.random.default_rng(5)
    data = {
        "unique_id": ["long1"] * 48 + ["long2"] * 48 + ["short"] * 15,
        "ds": list(range(48)) * 2 + list(range(15)),
        "y": rng.normal(size=111),
    }
    df = frame(engine, data)
    with pytest.raises(ValueError, match="non-finite features"):
        feature_coverage(df, df, seasonal_period=12)
    result = feature_coverage(df, df, seasonal_period=12, missing="drop")
    assert result.real_diagnostics.dropped_ids == ("short",)
    assert set(result.real_diagnostics.drop_reasons["short"]) == {
        "trend_strength",
        "seasonal_strength",
        "spike",
        "lumpiness",
        "max_level_shift",
        "max_var_shift",
    }


def test_reference_pca_matches_scikit_learn(engine):
    decomposition = pytest.importorskip("sklearn.decomposition")
    rng = np.random.default_rng(8)
    covariance = np.diag([4.0, 2.0, 1.0, 0.5])
    real_x = rng.multivariate_normal(np.zeros(4), covariance, size=60)
    syn_x = rng.multivariate_normal(np.ones(4), covariance, size=40)

    def as_frame(x, prefix):
        columns = {f"f{j}": x[:, j] for j in range(4)}
        return frame(
            engine, {"unique_id": [f"{prefix}{i}" for i in range(len(x))], **columns}
        )

    result = feature_coverage(
        as_frame(real_x, "r"), as_frame(syn_x, "s"), precomputed=True
    )
    mean, std = real_x.mean(axis=0), real_x.std(axis=0)
    pca = decomposition.PCA(2).fit((real_x - mean) / std)
    signs = np.sign(
        np.sum(pca.transform((real_x - mean) / std) * result.real_embedding, axis=0)
    )
    np.testing.assert_allclose(
        result.real_embedding, pca.transform((real_x - mean) / std) * signs, atol=1e-10
    )
    np.testing.assert_allclose(
        result.synthetic_embedding,
        pca.transform((syn_x - mean) / std) * signs,
        atol=1e-10,
    )
    np.testing.assert_allclose(result.explained_variance, pca.explained_variance_ratio_)
    # Row order of the reference does not change the fitted space.
    shuffled = rng.permutation(len(real_x))
    again = feature_coverage(
        as_frame(real_x[shuffled], "r"), as_frame(syn_x, "s"), precomputed=True
    )
    np.testing.assert_allclose(
        again.synthetic_embedding, result.synthetic_embedding, atol=1e-10
    )


def test_large_constant_columns_are_detected_exactly(engine):
    real = feature_frame(engine, constant=[123456.7] * 3)
    synthetic = feature_frame(engine, constant=[123456.7, 123457.7, 123456.7])
    result = feature_coverage(real, synthetic, precomputed=True)
    assert result.constant_features_dropped == ("constant",)
    assert result.synthetic_constant_feature_deviations == {"constant": 1}
    assert result.n_synthetic_out_of_range == 0


def test_feature_scaling_overflow_is_reported(engine):
    real = feature_frame(engine, (-1e308, 0.0, 1e308))
    with pytest.raises(ValueError, match="overflowed"):
        feature_coverage(real, real, precomputed=True)


def test_id_col_conflicts_and_empty_panels():
    with pytest.raises(ValueError, match="id_col"):
        compute_features(panel("pandas"), id_col="acf1")
    with pytest.raises(ValueError, match="id_col must not be a selected feature"):
        feature_coverage(
            feature_frame("pandas"),
            feature_frame("pandas"),
            precomputed=True,
            features=["unique_id"],
        )
    empty = pd.DataFrame({"unique_id": [], "ds": [], "y": []})
    with pytest.raises(ValueError, match="at least one series"):
        compute_features(empty)


def test_worker_counts_give_identical_features():
    rng = np.random.default_rng(4)
    data = {
        "unique_id": np.repeat([f"s{i}" for i in range(50)], 40),
        "ds": np.tile(np.arange(40), 50),
        "y": rng.normal(size=2000),
    }
    df = pd.DataFrame(data)
    single = compute_features(df, seasonal_period=4, n_jobs=1)
    for n_jobs in (4, -1):
        pd.testing.assert_frame_equal(
            compute_features(df, seasonal_period=4, n_jobs=n_jobs), single
        )
