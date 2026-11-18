import numpy as np
import pytest
from pydantic import ValidationError

from conformalforecast.methods.split import SplitConformal
from conformalforecast.methods.weighted_split import WeightedSplit, WeightedSplitForecaster
from conformalforecast.nonconformity.signed import SignedResidual
from conformalforecast.weighting import DensityRatio, PrecomputedWeights


def _shifted_panel(n, seed, loc):
    """Feature-dependent noise: variance grows with the single feature."""
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=loc, scale=1.0, size=(1, n, 1))
    noise = rng.normal(scale=1.0 + np.abs(x), size=(1, n, 1))
    preds = np.zeros((1, n, 1))
    truths = noise
    X = x[..., np.newaxis]
    return preds, truths, X


def test_uniform_weighting_matches_split_conformal():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(3, 100, 2))
    t = rng.normal(size=(3, 100, 2))
    a = SplitConformal(alpha=0.1).fit(p, t)
    b = WeightedSplit(alpha=0.1).fit(p, t)
    np.testing.assert_allclose(a.interval(p[:, :5]), b.interval(p[:, :5]))


def test_global_pooling_requires_scale_estimator():
    with pytest.raises(ValidationError, match="scale_estimator"):
        WeightedSplit(alpha=0.1, pooling="global")


def test_unknown_pooling_rejected():
    """A Literal field, so pydantic rejects the value before __init__ finishes."""
    with pytest.raises(ValidationError):
        WeightedSplit(alpha=0.1, pooling="pooled")


def test_interval_and_adapt_live_only_on_the_fitted_object():
    """Using them before fit is unrepresentable: the method has neither."""
    m = WeightedSplit(alpha=0.1)
    assert not hasattr(m, "interval")
    assert not hasattr(m, "adapt")


def test_adapt_with_no_arguments_raises():
    rng = np.random.default_rng(12)
    f = WeightedSplit(alpha=0.1).fit(rng.normal(size=(1, 100, 1)), rng.normal(size=(1, 100, 1)))
    with pytest.raises(ValueError, match="at least one"):
        f.adapt()


def test_adapt_target_x_without_a_weighting_function_raises():
    """Regression test: adapt(target_X=...) must not silently no-op when the
    bound weighting function cannot consume target features (the default
    UniformWeights) -- that would return a successful-looking adapted
    object with unchanged, unweighted intervals."""
    rng = np.random.default_rng(11)
    p = rng.normal(size=(1, 100, 1))
    t = rng.normal(size=(1, 100, 1))
    X = rng.normal(size=(1, 100, 1, 2))
    f = WeightedSplit(alpha=0.1).fit(p, t, X=X)
    with pytest.raises(ValueError, match="does not consume target features"):
        f.adapt(target_X=rng.normal(size=(1, 10, 1, 2)))
    assert f.weighting_.is_adapted_ is False


def test_adapt_target_x_with_precomputed_weights_raises():
    """PrecomputedWeights is adapted by construction yet ignores target
    features, so a guard reading is_adapted_ would let target_X through and
    discard it. The declaration must catch this shape of the same bug."""
    rng = np.random.default_rng(15)
    p = rng.normal(size=(1, 100, 1))
    t = rng.normal(size=(1, 100, 1))
    X = rng.normal(size=(1, 100, 1, 2))
    f = WeightedSplit(alpha=0.1).fit(p, t, X=X)
    with pytest.raises(ValueError, match="does not consume target features"):
        f.adapt(
            weighting=PrecomputedWeights(weights=np.ones((1, 100, 1))),
            target_X=rng.normal(size=(1, 10, 1, 2)),
        )


def _adapted_pair(seed=21):
    """A fitted WeightedSplit already adapted to a non-uniform weighting, plus
    a snapshot of the state that a later rejected adapt must not disturb."""
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(1, 100, 1))
    t = rng.normal(size=(1, 100, 1))
    X = rng.normal(size=(1, 100, 1, 2))
    weights = np.abs(rng.normal(size=(1, 100, 1))) + 0.5
    f = WeightedSplit(alpha=0.1, scale_estimator="mad").fit(p, t, X=X)
    f = f.adapt(weighting=PrecomputedWeights(weights=weights))
    return f, rng, f.weighting_, f.score_quantile_.copy()


def test_adapt_rejection_leaves_the_receiver_unchanged():
    """A rejected adapt must leave the receiver's adaptation intact: its
    weighting function and the quantile that describes it stay in step."""
    f, rng, previous, quantile = _adapted_pair()
    with pytest.raises(ValueError, match="does not consume target features"):
        f.adapt(
            weighting=PrecomputedWeights(weights=np.full((1, 100, 1), 2.0)),
            target_X=rng.normal(size=(1, 10, 1, 2)),
        )
    assert f.weighting_ is previous
    assert np.array_equal(f.score_quantile_, quantile)


def test_adapt_leaves_state_intact_when_the_weighting_fails_to_fit():
    """The estimator raising inside fit_target must not leave the receiver
    serving a source quantile through an unadapted DensityRatio."""
    pytest.importorskip("sklearn")
    f, rng, previous, quantile = _adapted_pair()

    class Exploding:
        def fit(self, X, y):
            raise RuntimeError("estimator exploded")

        def predict_proba(self, X):  # pragma: no cover - fit raises first
            raise RuntimeError

    with pytest.raises(RuntimeError, match="estimator exploded"):
        f.adapt(
            weighting=DensityRatio(estimator=Exploding()),
            target_X=rng.normal(size=(1, 10, 1, 2)),
        )
    assert f.weighting_ is previous
    assert np.array_equal(f.score_quantile_, quantile)


def test_adapt_without_a_scale_estimator_leaves_the_receiver_unchanged():
    """target_series is gated on scale_estimator; the rejection must come
    before the weighting argument reaches the receiver."""
    rng = np.random.default_rng(23)
    p = rng.normal(size=(1, 100, 1))
    t = rng.normal(size=(1, 100, 1))
    weights = np.abs(rng.normal(size=(1, 100, 1))) + 0.5
    f = WeightedSplit(alpha=0.1).fit(p, t)  # no scale_estimator
    f = f.adapt(weighting=PrecomputedWeights(weights=weights))
    previous, quantile = f.weighting_, f.score_quantile_.copy()
    with pytest.raises(ValueError, match="scale_estimator is None"):
        f.adapt(
            weighting=PrecomputedWeights(weights=np.ones((1, 100, 1))),
            target_series=rng.normal(size=(1, 50)),
        )
    assert f.weighting_ is previous
    assert np.array_equal(f.score_quantile_, quantile)


def test_adapt_rejected_by_recompute_leaves_scales_and_quantile_intact():
    """_recompute_quantile rejects a target panel whose series count differs
    under per_series pooling, after the scales have been computed -- the
    latest point a rejection can come, and still the receiver is unchanged."""
    f, rng, previous, quantile = _adapted_pair()
    with pytest.raises(ValueError, match="does not match the fitted n_series"):
        f.adapt(target_series=rng.normal(size=(3, 50)))
    assert f.weighting_ is previous
    assert f.target_scales_ is None
    assert np.array_equal(f.score_quantile_, quantile)


def test_dre_weights_restore_coverage_under_covariate_shift():
    """The load-bearing test: weights must fix coverage that uniform weights break."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    cal_p, cal_t, cal_X = _shifted_panel(2000, 0, loc=0.0)
    test_p, test_t, test_X = _shifted_panel(4000, 1, loc=2.5)

    def coverage(fitted):
        iv = fitted.interval(test_p, X=test_X)
        return float(((test_t >= iv[..., 0]) & (test_t <= iv[..., 1])).mean())

    plain = WeightedSplit(alpha=0.1).fit(cal_p, cal_t, X=cal_X)
    uniform_cov = coverage(plain)

    weighted = plain.adapt(
        # clip_quantile=None: this test measures one property - that density-ratio
        # weights restore coverage under shift. Clipping is a separate bias/variance
        # safeguard, and at the 0.99 default it binds on over half these points
        # (median test weight saturates at the clip), costing ~10 coverage points
        # and masking the effect under test.
        weighting=DensityRatio(
            estimator=LogisticRegression(max_iter=1000), cv=3, clip_quantile=None
        ),
        target_X=test_X,
    )
    weighted_cov = coverage(weighted)

    # Negative control: without weights the shift breaks coverage.
    assert uniform_cov < 0.87
    # With density-ratio weights, coverage returns toward nominal.
    assert weighted_cov > uniform_cov + 0.02
    assert abs(weighted_cov - 0.9) < 0.05


def test_adapt_is_idempotent():
    rng = np.random.default_rng(2)
    p = rng.normal(size=(2, 200, 1))
    t = rng.normal(size=(2, 200, 1))
    hist = np.cumsum(rng.normal(size=(2, 500)), axis=1)
    f = WeightedSplit(alpha=0.1, scale_estimator="std").fit(p, t)
    once = f.adapt(target_series=hist)
    # Adapting the adapted object again recomputes from source scores rather
    # than compounding the scale ratio.
    twice = once.adapt(target_series=hist)
    np.testing.assert_array_equal(once.interval(p[:, :3]), twice.interval(p[:, :3]))


def test_per_series_adapt_rejects_a_mismatched_target_panel():
    """Under per-series pooling the scale ratio is elementwise, so a target
    panel of a different width used to broadcast instead of raising."""
    rng = np.random.default_rng(11)
    p = rng.normal(size=(3, 200, 1))
    t = rng.normal(size=(3, 200, 1))
    f = WeightedSplit(alpha=0.1, scale_estimator="std").fit(p, t)
    good = np.cumsum(rng.normal(size=(3, 500)), axis=1)
    f = f.adapt(target_series=good)
    before = f.interval(p[:, :3]).copy()
    ratio_before = f.scale_ratio_.copy()

    for n_target in (5, 1):  # 1 is the one that silently broadcast
        with pytest.raises(ValueError, match="target_series n_series"):
            f.adapt(target_series=np.cumsum(rng.normal(size=(n_target, 500)), axis=1))
        # A rejected adapt must leave the receiver's adaptation intact.
        np.testing.assert_allclose(f.target_scales_, f.scale_ratio_ * f.source_scales_)
        np.testing.assert_array_equal(f.scale_ratio_, ratio_before)
        np.testing.assert_array_equal(f.interval(p[:, :3]), before)

    # Still adaptable.
    np.testing.assert_array_equal(f.adapt(target_series=good).interval(p[:, :3]), before)


def test_mismatched_x_sample_axis_raises():
    """On the adapted per-test-point path the test weights, not the
    predictions, used to shape the returned interval."""
    p, t, X = _shifted_panel(200, seed=12, loc=0.0)
    f = WeightedSplit(alpha=0.1).fit(p, t, X=X)
    f = f.adapt(weighting=PrecomputedWeights(weights=np.ones_like(t)))
    with pytest.raises(ValueError, match="test weights per series"):
        f.interval(p[:, :1], X=X)  # 1 prediction sample, 200 rows of X


def test_scale_alignment_widens_for_a_noisier_target():
    rng = np.random.default_rng(3)
    p = rng.normal(size=(1, 300, 1))
    t = rng.normal(size=(1, 300, 1))
    f = WeightedSplit(alpha=0.1, scale_estimator="std").fit(p, t)
    before = f.interval(np.zeros((1, 1, 1)))
    noisy = np.cumsum(rng.normal(scale=5.0, size=(1, 1000)), axis=1)
    after = f.adapt(target_series=noisy).interval(np.zeros((1, 1, 1)))
    width_before = before[0, 0, 0, 1] - before[0, 0, 0, 0]
    width_after = after[0, 0, 0, 1] - after[0, 0, 0, 0]
    assert width_after > width_before


def _proportional_panel(n_windows=400, horizon=2, factor=6.0, seed=101):
    """Two source series drawn from one set of steps, the second scaled by
    ``factor``. Scale normalization is then exact rather than statistical, so
    global pooling can be asserted against per-series pooling to machine
    precision instead of within a coverage tolerance."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(1, n_windows, horizon))
    truths = np.concatenate([base, factor * base], axis=0)
    return np.zeros_like(truths), truths


def test_global_pooling_serves_a_series_absent_from_source():
    rng = np.random.default_rng(4)
    p = rng.normal(size=(4, 200, 2))
    t = rng.normal(size=(4, 200, 2))
    f = WeightedSplit(alpha=0.1, pooling="global", scale_estimator="std").fit(p, t)
    # A target panel with a different series count than the source.
    target_hist = np.cumsum(rng.normal(size=(7, 400)), axis=1)
    m = f.adapt(target_series=target_hist)
    iv = m.interval(np.zeros((7, 1, 2)))
    assert iv.shape == (7, 1, 2, 2)
    assert np.all(np.isfinite(iv))
    # Widths must track each target series' own scale, not merely be finite.
    widths = iv[:, 0, 0, 1] - iv[:, 0, 0, 0]
    np.testing.assert_allclose(
        widths / widths[0], m.target_scales_ / m.target_scales_[0], rtol=1e-12
    )


def test_global_pooling_normalizes_by_the_source_scale_before_pooling():
    """Normalize, pool, re-scale must recover what each series would have got
    on its own -- exactly, for series that differ only by a constant factor.
    Pooling the raw scores instead leaves the noisier series dominating the
    quantile both series are served from, which is off by several fold."""
    p, t = _proportional_panel()
    pooled = WeightedSplit(alpha=0.1, pooling="global", scale_estimator="mad").fit(p, t)
    per_series = WeightedSplit(alpha=0.1, pooling="per_series", scale_estimator="mad").fit(p, t)
    np.testing.assert_allclose(pooled.score_quantile_, per_series.score_quantile_, rtol=1e-12)
    # The point of normalizing: the pooled quantile is in normalized units, an
    # order of magnitude below the scale of the wider series.
    assert pooled.base_quantile_.max() < per_series.score_quantile_[1].min() / 4


def test_global_pooling_adapt_scales_each_target_row_by_its_own_scale():
    """adapt(target_series=...) widens the pooled quantile to one row per
    target series through the target scale. Leaving the ratio at ones would
    serve every target series the same normalized-unit quantile."""
    p, t = _proportional_panel()
    f = WeightedSplit(alpha=0.1, pooling="global", scale_estimator="mad").fit(p, t)
    rng = np.random.default_rng(102)
    steps = rng.normal(size=(1, 600))
    # Three target series at 1x, 4x and 20x one common walk.
    factors = np.array([1.0, 4.0, 20.0])
    target_hist = np.cumsum(steps * factors[:, np.newaxis], axis=1)
    m = f.adapt(target_series=target_hist)

    np.testing.assert_allclose(m.scale_ratio_, m.target_scales_, rtol=1e-12)
    q = m.score_quantile_
    assert q.shape == (3, 2)
    np.testing.assert_allclose(q / q[0], np.tile(factors[:, np.newaxis], (1, 2)), rtol=1e-12)
    # Every row is the pooled quantile times its own scale, so no row is left
    # in the normalized units base_quantile_ carries.
    np.testing.assert_allclose(q, m.base_quantile_ * m.target_scales_[:, np.newaxis], rtol=1e-12)
    assert q[2, 0] > 10.0 * m.base_quantile_[0, 0]


def test_per_test_point_path_applies_the_scale_ratio():
    """With uniform weights the per-test-point quantile equals the plain
    weighted one, so the X and no-X branches must agree. They only do if the
    X branch applies scale_ratio_ too -- otherwise it serves source-unit
    intervals while the no-X branch serves target-unit ones."""
    p, t = _proportional_panel()
    rng = np.random.default_rng(103)
    f = WeightedSplit(alpha=0.1, pooling="per_series", scale_estimator="mad").fit(
        p, t, X=rng.normal(size=(2, 400, 2, 1))
    )
    steps = rng.normal(size=(1, 600))
    # A target three times the source scale, so a missing multiply is a 3x miss.
    target_hist = np.cumsum(3.0 * np.concatenate([steps, 6.0 * steps], axis=0), axis=1)
    m = f.adapt(
        weighting=PrecomputedWeights(weights=np.ones((2, 400, 2))),
        target_series=target_hist,
    )
    assert m.scale_ratio_ is not None
    preds = np.zeros((2, 5, 2))
    plain = m.interval(preds)
    reweighted = m.interval(preds, X=rng.normal(size=(2, 5, 2, 1)))
    np.testing.assert_allclose(reweighted, plain, rtol=1e-12)
    # And that shared width really is the scaled one, not the source width.
    unscaled = WeightedSplit(alpha=0.1, pooling="per_series").fit(p, t)
    scaled_width = reweighted[:, 0, 0, 1] - reweighted[:, 0, 0, 0]
    source_iv = unscaled.interval(preds)
    source_width = source_iv[:, 0, 0, 1] - source_iv[:, 0, 0, 0]
    np.testing.assert_allclose(scaled_width / source_width, m.scale_ratio_, rtol=1e-12)


def test_global_pooling_rejects_a_narrower_target_panel():
    """A multi-series source fitted under global pooling has one quantile row
    per source series until adapt(target_series=...) widens it, so a narrower
    forecast must raise rather than broadcast the quantile rows against the
    wrong predictions."""
    rng = np.random.default_rng(16)
    m = WeightedSplit(alpha=0.1, pooling="global", scale_estimator="std").fit(
        rng.normal(size=(4, 200, 2)), rng.normal(size=(4, 200, 2))
    )
    with pytest.raises(ValueError, match="does not match the fitted quantile"):
        m.interval(np.zeros((1, 3, 2)))


def test_global_pooling_rejects_a_wider_target_panel():
    """The same named error, not a raw numpy broadcast failure."""
    rng = np.random.default_rng(17)
    m = WeightedSplit(alpha=0.1, pooling="global", scale_estimator="std").fit(
        rng.normal(size=(4, 200, 2)), rng.normal(size=(4, 200, 2))
    )
    with pytest.raises(ValueError, match="does not match the fitted quantile"):
        m.interval(np.zeros((7, 3, 2)))


def test_global_pooling_preserves_float32():
    """The global branch divides by the float64 source scales; no upcast."""
    rng = np.random.default_rng(18)
    p = rng.normal(size=(4, 200, 2)).astype(np.float32)
    t = rng.normal(size=(4, 200, 2)).astype(np.float32)
    m = WeightedSplit(alpha=0.1, pooling="global", scale_estimator="std").fit(p, t)
    assert m.base_quantile_.dtype == np.float32
    assert m.score_quantile_.dtype == np.float32
    assert m.interval(p[:, :2]).dtype == np.float32


def test_nan_masking_equals_dense():
    rng = np.random.default_rng(5)
    dense_p = rng.normal(size=(1, 120, 2))
    dense_t = rng.normal(size=(1, 120, 2))
    padded_p = np.concatenate([np.full((1, 20, 2), np.nan), dense_p], axis=1)
    padded_t = np.concatenate([np.full((1, 20, 2), np.nan), dense_t], axis=1)
    a = WeightedSplit(alpha=0.1).fit(dense_p, dense_t)
    b = WeightedSplit(alpha=0.1).fit(padded_p, padded_t)
    np.testing.assert_allclose(a.score_quantile_, b.score_quantile_)


def test_precomputed_weights_are_used():
    """Weights supplied at adapt() with no target data at all."""
    rng = np.random.default_rng(6)
    p = rng.normal(size=(1, 100, 1))
    t = rng.normal(size=(1, 100, 1))
    w = np.ones((1, 100, 1))
    w[0, :50, 0] = 0.0  # ignore the first half entirely
    m = WeightedSplit(alpha=0.1).fit(p, t).adapt(weighting=PrecomputedWeights(weights=w))
    ref = WeightedSplit(alpha=0.1).fit(p[:, 50:], t[:, 50:])
    np.testing.assert_allclose(m.score_quantile_, ref.score_quantile_)


@pytest.mark.parametrize("n_series", [1, 5])
@pytest.mark.parametrize("horizon", [1, 12])
def test_shapes(n_series, horizon):
    rng = np.random.default_rng(7)
    p = rng.normal(size=(n_series, 100, horizon))
    t = rng.normal(size=(n_series, 100, horizon))
    m = WeightedSplit(alpha=0.1).fit(p, t)
    assert m.score_quantile_.shape == (n_series, horizon)
    assert m.interval(p[:, :3]).shape == (n_series, 3, horizon, 2)


def test_float32_preserved():
    rng = np.random.default_rng(8)
    p = rng.normal(size=(1, 100, 1)).astype(np.float32)
    t = rng.normal(size=(1, 100, 1)).astype(np.float32)
    m = WeightedSplit(alpha=0.1).fit(p, t)
    assert m.interval(p[:, :2]).dtype == np.float32


def test_determinism():
    def build():
        rng = np.random.default_rng(9)
        p = rng.normal(size=(2, 100, 2))
        t = rng.normal(size=(2, 100, 2))
        return WeightedSplit(alpha=0.1).fit(p, t).interval(p[:, :4])

    np.testing.assert_array_equal(build(), build())


def test_asymmetric_score_rejected_at_construction():
    """An asymmetric score would be silently ignored by the per-test-point path.

    The score is config, so the SCORE_TYPE gate rejects it when the method is
    constructed -- before any data is seen.
    """
    with pytest.raises(ValidationError, match="AbsoluteResidual"):
        WeightedSplit(alpha=0.1, score=SignedResidual())


def test_float32_preserved_on_the_adapted_fast_path():
    """The per-test-point branch multiplies by a float64 scale ratio; no upcast."""
    rng = np.random.default_rng(10)
    p = rng.normal(size=(1, 100, 1)).astype(np.float32)
    t = rng.normal(size=(1, 100, 1)).astype(np.float32)
    X = rng.normal(size=(1, 100, 1, 2)).astype(np.float32)
    m = WeightedSplit(alpha=0.1, scale_estimator="std").fit(p, t, X=X)
    m = m.adapt(
        weighting=PrecomputedWeights(weights=np.ones((1, 100, 1))),
        target_series=np.cumsum(rng.normal(size=(1, 400)), axis=1),
    )
    assert m.weighting_.is_adapted_
    assert m.scale_ratio_ is not None
    iv = m.interval(p[:, :2], X=X[:, :2])
    assert iv.dtype == np.float32


def test_schema_is_generatable_and_estimator_free():
    """Construction config is scalar-only apart from the tagged score, so the
    LLM-facing schema exists.

    An arbitrary-type field (a classifier, an array) reaches pydantic as an
    IsInstanceSchema, which has no JSON schema at all -- model_json_schema()
    would raise (that call not raising is itself the primary assertion here).
    Keeping the weight function out of __init__ is what makes this method
    describable. A nested submodel field (as ``weighting: WeightFunction``
    would be) additionally shows up as a "$defs" entry, which this config
    never produces -- the score's schema is inlined -- and that check is
    substring-safe, unlike checking for "estimator", which false-positives on
    the ``scale_estimator`` field name itself.
    """
    schema = WeightedSplit.model_json_schema()
    assert set(schema["properties"]) == {"alpha", "seed", "score", "pooling", "scale_estimator"}
    assert "$defs" not in schema


def test_fitted_and_adapted_state_lives_on_the_fitted_object():
    """The method carries no fitted state at all, so its schema cannot leak it."""
    rng = np.random.default_rng(14)
    f = WeightedSplit(alpha=0.1).fit(rng.normal(size=(1, 100, 1)), rng.normal(size=(1, 100, 1)))
    f = f.adapt(weighting=PrecomputedWeights(weights=np.ones((1, 100, 1))))
    state = {"weighting_", "scores_", "score_quantile_", "X_calibration_"}
    assert state <= set(type(f).model_fields)
    assert state.isdisjoint(WeightedSplit.model_fields)
    assert state.isdisjoint(WeightedSplit.model_json_schema()["properties"])


def test_adapt_with_density_ratio_on_a_ragged_panel():
    """Left-NaN padding is a first-class input, but the padded feature rows
    used to reach the classifier and fail inside scikit-learn. End-to-end
    through the method, which is where the shapes are actually assembled."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(31)
    p = rng.normal(size=(2, 80, 1))
    t = rng.normal(size=(2, 80, 1))
    X = rng.normal(size=(2, 80, 1, 2))
    p[0, :20] = np.nan
    t[0, :20] = np.nan
    X[0, :20] = np.nan

    f = WeightedSplit(alpha=0.1).fit(p, t, X=X)
    m = f.adapt(
        weighting=DensityRatio(estimator=LogisticRegression(max_iter=1000)),
        target_X=rng.normal(loc=1.0, size=(2, 30, 1, 2)),
    )
    weights = m.weighting_.calibration_weights()
    assert np.all(weights[0, :20] == 0.0)
    assert np.all(weights[0, 20:] > 0.0)
    # The summary describes real windows, so the padding does not drag min to 0.
    assert m.calibration_.diagnostics["weight_summary"]["min"] > 0.0
    iv = m.interval(p[:, 20:25], X=X[:, 20:25])
    assert iv.shape == (2, 5, 1, 2)
    assert np.all(np.isfinite(iv))


def test_ragged_adapt_matches_adapting_each_series_dense():
    """The NaN-masking-equals-dense invariant, extended across adapt: a padded
    series must get the quantile it would get from its own dense sub-array."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(37)
    p = rng.normal(size=(1, 80, 1))
    t = rng.normal(size=(1, 80, 1))
    X = rng.normal(size=(1, 80, 1, 2))
    target_X = rng.normal(loc=1.0, size=(1, 30, 1, 2))

    padded_p, padded_t, padded_X = p.copy(), t.copy(), X.copy()
    padded_p[0, :20] = np.nan
    padded_t[0, :20] = np.nan
    padded_X[0, :20] = np.nan

    def quantile(preds, truths, feats):
        f = WeightedSplit(alpha=0.1).fit(preds, truths, X=feats)
        return f.adapt(
            weighting=DensityRatio(estimator=LogisticRegression(max_iter=1000), cv=3),
            target_X=target_X,
        ).score_quantile_

    np.testing.assert_allclose(
        quantile(padded_p, padded_t, padded_X),
        quantile(p[:, 20:], t[:, 20:], X[:, 20:]),
    )


class TestAdaptReturnsANewObject:
    """``adapt`` is "fitted -> new fitted": the receiver is never modified."""

    def _fitted(self, seed: int = 41) -> tuple[WeightedSplitForecaster, np.random.Generator]:
        rng = np.random.default_rng(seed)
        p = rng.normal(size=(2, 120, 1))
        t = rng.normal(size=(2, 120, 1))
        return WeightedSplit(alpha=0.1, scale_estimator="std").fit(p, t), rng

    def test_returns_a_new_object_of_the_same_class(self):
        f, rng = self._fitted()
        adapted = f.adapt(target_series=np.cumsum(rng.normal(scale=4.0, size=(2, 400)), axis=1))
        assert adapted is not f
        assert type(adapted) is WeightedSplitForecaster
        assert adapted.calibration_.diagnostics["path"] == "adapt"
        assert not np.array_equal(adapted.score_quantile_, f.score_quantile_)

    def test_receiver_is_unchanged(self):
        f, rng = self._fitted()
        quantile, weighting, calibration = f.score_quantile_.copy(), f.weighting_, f.calibration_
        weights = np.abs(rng.normal(size=(2, 120, 1))) + 0.5
        f.adapt(
            weighting=PrecomputedWeights(weights=weights),
            target_series=np.cumsum(rng.normal(scale=4.0, size=(2, 400)), axis=1),
        )
        np.testing.assert_array_equal(f.score_quantile_, quantile)
        assert f.weighting_ is weighting
        assert f.target_scales_ is None
        assert f.scale_ratio_ is None
        assert f.calibration_ is calibration
        assert "path" not in f.calibration_.diagnostics

    def test_weighting_is_shared_with_neither_receiver_nor_caller(self):
        f, rng = self._fitted()
        passed = PrecomputedWeights(weights=np.abs(rng.normal(size=(2, 120, 1))) + 0.5)
        snapshot = passed.model_copy(deep=True)
        adapted = f.adapt(weighting=passed)
        assert adapted.weighting_ is not passed
        assert adapted.weighting_ is not f.weighting_
        # fit_source ran on the copy, so the caller's instance is untouched.
        assert passed == snapshot

    def test_reusing_the_current_weighting_copies_it(self):
        """With weighting=None the receiver's weighting is the base -- still copied."""
        f, rng = self._fitted()
        adapted = f.adapt(target_series=np.cumsum(rng.normal(size=(2, 400)), axis=1))
        assert adapted.weighting_ is not f.weighting_

    def test_is_idempotent_on_the_same_receiver(self):
        f, rng = self._fitted()
        hist = np.cumsum(rng.normal(scale=3.0, size=(2, 400)), axis=1)
        weights = np.abs(rng.normal(size=(2, 120, 1))) + 0.5
        first = f.adapt(weighting=PrecomputedWeights(weights=weights), target_series=hist)
        second = f.adapt(weighting=PrecomputedWeights(weights=weights), target_series=hist)
        np.testing.assert_array_equal(first.score_quantile_, second.score_quantile_)
        np.testing.assert_array_equal(first.scale_ratio_, second.scale_ratio_)
