import warnings

import numpy as np
import pytest

from conformalforecast.weighting import DensityRatio


class _ConstantProba:
    """Classifier returning a fixed P(target|x), for deterministic assertions."""

    def __init__(self, p=0.5):
        self.p = p
        self.n_fit_calls = 0

    def fit(self, X, y):
        self.n_fit_calls += 1
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def _exog(n_series, n_obs, horizon, n_features, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_series, n_obs, horizon, n_features))


def test_odds_ratio_of_half_is_one():
    d = DensityRatio(estimator=_ConstantProba(0.5), cv=2)
    d.fit_source(_exog(2, 10, 3, 2, 0), (2, 10, 3))
    d.fit_target(_exog(2, 10, 3, 2, 1))
    np.testing.assert_allclose(d.calibration_weights(), np.ones((2, 10, 3)))


def test_odds_ratio_of_two_thirds_is_two():
    d = DensityRatio(estimator=_ConstantProba(2 / 3), cv=2)
    d.fit_source(_exog(2, 10, 3, 2, 0), (2, 10, 3))
    d.fit_target(_exog(2, 10, 3, 2, 1))
    np.testing.assert_allclose(d.calibration_weights(), np.full((2, 10, 3), 2.0))


def test_unadapted_returns_uniform_weights():
    d = DensityRatio(estimator=_ConstantProba(0.9), cv=2)
    d.fit_source(_exog(2, 10, 3, 2, 0), (2, 10, 3))
    assert d.is_adapted_ is False
    np.testing.assert_array_equal(d.calibration_weights(), np.ones((2, 10, 3)))


def test_cross_fitting_fits_cv_plus_one_models():
    est = _ConstantProba(0.5)
    d = DensityRatio(estimator=est, cv=4)
    d.fit_source(_exog(2, 10, 3, 2, 0), (2, 10, 3))
    d.fit_target(_exog(2, 10, 3, 2, 1))
    # cv out-of-fold models plus one final model fitted on all data.
    assert est.n_fit_calls == 5


def test_cv_below_two_disables_cross_fitting():
    est = _ConstantProba(0.5)
    d = DensityRatio(estimator=est, cv=1)
    d.fit_source(_exog(2, 10, 3, 2, 0), (2, 10, 3))
    d.fit_target(_exog(2, 10, 3, 2, 1))
    assert est.n_fit_calls == 1


def test_clipping_bounds_extreme_weights():
    class _Escalating:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            p = np.linspace(0.5, 0.999, len(X))
            return np.column_stack([1 - p, p])

    d = DensityRatio(estimator=_Escalating(), cv=1, clip_quantile=0.5)
    d.fit_source(_exog(1, 20, 1, 2, 0), (1, 20, 1))
    d.fit_target(_exog(1, 20, 1, 2, 1))
    w = d.calibration_weights()
    assert w.max() < 5.0


def test_requires_exog_features():
    d = DensityRatio(estimator=_ConstantProba(), cv=2)
    d.fit_source(None, (2, 10, 3))
    with pytest.raises(ValueError, match="exogenous features"):
        d.fit_target(_exog(2, 10, 3, 2, 1))


def test_feature_count_must_match():
    d = DensityRatio(estimator=_ConstantProba(), cv=2)
    d.fit_source(_exog(2, 10, 3, 2, 0), (2, 10, 3))
    with pytest.raises(ValueError, match="n_features"):
        d.fit_target(_exog(2, 10, 3, 5, 1))


def test_test_weights_shape_follows_exog():
    d = DensityRatio(estimator=_ConstantProba(0.5), cv=2)
    d.fit_source(_exog(2, 10, 3, 2, 0), (2, 10, 3))
    d.fit_target(_exog(2, 10, 3, 2, 1))
    w = d.test_weights(_exog(2, 7, 3, 2, 2))
    assert w.shape == (2, 7, 3)


def test_determinism():
    def build():
        d = DensityRatio(estimator=_ConstantProba(0.7), cv=3, seed=0)
        d.fit_source(_exog(2, 12, 2, 3, 0), (2, 12, 2))
        d.fit_target(_exog(2, 12, 2, 3, 1))
        return d.calibration_weights()

    np.testing.assert_array_equal(build(), build())


def test_sklearn_logistic_regression_plugs_in_unwrapped():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    source = rng.normal(loc=0.0, size=(2, 40, 2, 3))
    target = rng.normal(loc=2.0, size=(2, 40, 2, 3))
    d = DensityRatio(estimator=LogisticRegression(max_iter=1000), cv=3)
    d.fit_source(source, (2, 40, 2))
    d.fit_target(target)
    w = d.calibration_weights()
    assert w.shape == (2, 40, 2)
    assert np.all(np.isfinite(w))
    assert np.all(w >= 0)


def test_sklearn_gradient_boosting_plugs_in_unwrapped():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.default_rng(1)
    source = rng.normal(loc=0.0, size=(1, 40, 1, 2))
    target = rng.normal(loc=3.0, size=(1, 40, 1, 2))
    d = DensityRatio(estimator=GradientBoostingClassifier(n_estimators=10), cv=2)
    d.fit_source(source, (1, 40, 1))
    d.fit_target(target)
    assert np.all(np.isfinite(d.calibration_weights()))


def test_cross_fit_folds_capped_at_the_rarer_class():
    """With fewer target rows than ``cv``, one fold used to hold the whole
    class and leave its complement single-class."""
    d = DensityRatio(estimator=_ConstantProba(0.5), cv=5)
    d.fit_source(_exog(1, 10, 1, 2, 0), (1, 10, 1))
    with pytest.warns(UserWarning, match="capped to 1 folds"):
        d.fit_target(_exog(1, 1, 1, 2, 1))
    assert d.is_adapted_ is True
    assert d.calibration_weights().shape == (1, 10, 1)


def test_cross_fit_folds_capped_but_still_cross_fitted():
    """Between 2 and cv folds the cross-fit survives, with fewer folds."""
    d = DensityRatio(estimator=_ConstantProba(0.5), cv=5)
    d.fit_source(_exog(1, 10, 1, 2, 0), (1, 10, 1))
    with pytest.warns(UserWarning, match="capped to 2 folds"):
        d.fit_target(_exog(1, 2, 1, 2, 1))
    # 2 cross-fit folds plus the final all-data model.
    assert d.estimator.n_fit_calls == 3


def test_cross_fit_uses_every_fold_when_the_panel_is_large_enough():
    d = DensityRatio(estimator=_ConstantProba(0.5), cv=3)
    d.fit_source(_exog(1, 9, 1, 2, 0), (1, 9, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d.fit_target(_exog(1, 9, 1, 2, 1))
    # 3 cross-fit folds plus the final all-data model.
    assert d.estimator.n_fit_calls == 4


def _ragged_exog(n_series, n_obs, horizon, n_features, seed, n_pad):
    """Source features for a panel whose first series lacks its earliest
    windows, so those cells are NaN — the package's left-padding contract."""
    X = _exog(n_series, n_obs, horizon, n_features, seed)
    X[0, :n_pad] = np.nan
    valid = np.ones((n_series, n_obs, horizon), dtype=bool)
    valid[0, :n_pad] = False
    return X, valid


def test_ragged_source_features_are_dropped_not_handed_to_the_estimator():
    """A ragged panel pads absent windows with NaN. Those rows used to go
    straight into estimator.fit, which failed inside the third-party library
    on a first-class supported input shape."""
    X, valid = _ragged_exog(2, 20, 1, 2, 0, n_pad=5)
    d = DensityRatio(estimator=_ConstantProba(0.6), cv=3)
    d.fit_source(X, (2, 20, 1), valid=valid)
    d.fit_target(_exog(2, 20, 1, 2, 1))
    w = d.calibration_weights()
    assert w.shape == (2, 20, 1)
    assert np.all(np.isfinite(w))
    # Padded cells carry weight 0; the weighted quantile zeroes them anyway.
    assert np.all(w[0, :5] == 0.0)
    assert np.all(w[0, 5:] > 0.0)
    assert np.all(w[1] > 0.0)


def test_ragged_source_only_fits_the_real_windows():
    """The dropped rows must not reach the estimator at all: the classifier
    sees 35 source rows, not the 40 the padded block contains."""
    seen = []

    class _Recording(_ConstantProba):
        def fit(self, X, y):
            seen.append((len(X), int(np.sum(y == 0))))
            return super().fit(X, y)

    X, valid = _ragged_exog(2, 20, 1, 2, 0, n_pad=5)
    d = DensityRatio(estimator=_Recording(0.5), cv=0)
    d.fit_source(X, (2, 20, 1), valid=valid)
    d.fit_target(_exog(1, 10, 1, 2, 1))
    assert seen == [(45, 35)]  # 35 live source rows + 10 target rows


def test_sklearn_accepts_a_ragged_source_panel():
    """The regression the reviewers hit was sklearn's own 'Input X contains
    NaN', so pin it against a real estimator rather than only the stub."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    X, valid = _ragged_exog(2, 40, 2, 3, 0, n_pad=10)
    d = DensityRatio(estimator=LogisticRegression(max_iter=1000), cv=3)
    d.fit_source(X, (2, 40, 2), valid=valid)
    d.fit_target(_exog(2, 40, 2, 3, 1) + 2.0)
    assert np.all(np.isfinite(d.calibration_weights()))


def test_non_finite_feature_on_a_live_window_is_rejected():
    """NaN where the window does exist is bad data, not padding."""
    X, valid = _ragged_exog(2, 20, 1, 2, 0, n_pad=5)
    X[1, 7, 0, 0] = np.nan  # series 1 has every window
    d = DensityRatio(estimator=_ConstantProba(0.5), cv=0)
    d.fit_source(X, (2, 20, 1), valid=valid)
    with pytest.raises(ValueError, match="interior or trailing NaN is an error"):
        d.fit_target(_exog(2, 20, 1, 2, 1))


def test_non_finite_source_features_rejected_without_a_valid_mask():
    """With no mask every cell counts as a real window, so NaN is an error
    rather than being silently dropped. The message must name the missing mask,
    because a caller with a genuinely ragged panel who simply forgot to pass it
    lands here and would otherwise be told their data is bad."""
    d = DensityRatio(estimator=_ConstantProba(0.5), cv=0)
    X = _exog(1, 20, 1, 2, 0)
    X[0, 3] = np.nan
    d.fit_source(X, (1, 20, 1))
    with pytest.raises(ValueError, match="no valid mask was supplied to fit_source"):
        d.fit_target(_exog(1, 20, 1, 2, 1))


def test_all_source_windows_absent_is_rejected():
    d = DensityRatio(estimator=_ConstantProba(0.5), cv=0)
    X = np.full((1, 10, 1, 2), np.nan)
    d.fit_source(X, (1, 10, 1), valid=np.zeros((1, 10, 1), dtype=bool))
    with pytest.raises(ValueError, match="no source exogenous rows remain"):
        d.fit_target(_exog(1, 10, 1, 2, 1))


def test_non_finite_target_features_rejected():
    """Every target sample is a step to be predicted, so NaN means nothing
    there and must not be quietly labelled as a target row."""
    d = DensityRatio(estimator=_ConstantProba(0.5), cv=0)
    d.fit_source(_exog(1, 20, 1, 2, 0), (1, 20, 1))
    target = _exog(1, 10, 1, 2, 1)
    target[0, 4] = np.nan
    with pytest.raises(ValueError, match="target exogenous features contain non-finite"):
        d.fit_target(target)
    assert d.is_adapted_ is False


def test_non_finite_test_features_rejected():
    d = DensityRatio(estimator=_ConstantProba(0.6), cv=0)
    d.fit_source(_exog(1, 20, 1, 2, 0), (1, 20, 1))
    d.fit_target(_exog(1, 20, 1, 2, 1))
    X = _exog(1, 5, 1, 2, 2)
    X[0, 2] = np.nan
    with pytest.raises(ValueError, match="test exogenous features contain non-finite"):
        d.test_weights(X)
