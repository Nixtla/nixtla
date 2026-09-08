"""Arguments that used to reach the kernels and read or write out of bounds.

Every case here was confirmed with AddressSanitizer or UBSan before the checks
it exercises were added.
"""

import numpy as np
import pytest

from coreforecast.differences import num_seas_diffs
from coreforecast.grouped_array import GroupedArray
from coreforecast.lag_transforms import ExpandingMean, Lag
from coreforecast.scalers import LocalBoxCoxScaler, boxcox_lambda
from coreforecast.seasonal import find_season_length


@pytest.fixture
def ga():
    return GroupedArray(np.arange(10.0), np.array([0, 5, 10], dtype=np.int32))


class TestK:
    @pytest.mark.parametrize(
        "call",
        [
            lambda ga, k: ga._index_from_end(k),
            lambda ga, k: ga._head(k),
            lambda ga, k: ga._tail(k),
        ],
        ids=["index_from_end", "head", "tail"],
    )
    def test_rejects_a_negative_k(self, ga, call):
        # k counts back from the end of a group, so -1 read one past it
        with pytest.raises(ValueError, match="k must be non-negative"):
            call(ga, -1)

    @pytest.mark.parametrize(
        "call", [lambda ga: ga._head(0), lambda ga: ga._tail(0)], ids=["head", "tail"]
    )
    def test_zero_k_is_empty(self, ga, call):
        assert call(ga).size == 0

    def test_lag_zero_cannot_update(self, ga):
        # an update consumes the value that isn't in ga yet, so lag 0 has
        # nothing to read and used to index one past the group
        with pytest.raises(ValueError, match="lag must be greater than 0"):
            Lag(0).update(ga)

    def test_lag_zero_transforms_but_cannot_update(self, ga):
        tfm = ExpandingMean(0)
        tfm.transform(ga)  # an un-lagged transform is legitimate
        with pytest.raises(ValueError, match="lag must be greater than 0"):
            tfm.update(ga)
        # the rejected update must not have advanced the counts
        np.testing.assert_array_equal(tfm.stats_[:, 0], [5.0, 5.0])


class TestPeriodsValidation:
    @pytest.mark.parametrize("size", [1, 200], ids=["undersized", "oversized"])
    def test_rejects_a_periods_that_is_not_one_per_group(self, ga, size):
        # the copy loop runs over periods, so an oversized one wrote past the
        # buffer it fills and an undersized one left the rest uninitialised
        with pytest.raises(ValueError, match="periods must have one element per group"):
            ga._num_seas_diffs_periods(1, np.full(size, 2.0))

    def test_matches_the_scalar_season_length_path(self):
        x = np.tile(np.arange(12.0), 5) + np.arange(60.0)
        ga = GroupedArray(np.hstack([x, x]), np.array([0, 60, 120], dtype=np.int32))
        np.testing.assert_array_equal(
            ga._num_seas_diffs_periods(1, np.full(2, 12.0)), ga._num_seas_diffs(12, 1)
        )


class TestStatsValidation:
    @pytest.mark.parametrize(
        "call",
        [
            lambda ga, stats: ga._scaler_transform(stats),
            lambda ga, stats: ga._scaler_inverse_transform(stats),
            lambda ga, stats: ga._boxcox(stats),
            lambda ga, stats: ga._inv_boxcox(stats),
        ],
        ids=["scaler_transform", "scaler_inverse_transform", "boxcox", "inv_boxcox"],
    )
    @pytest.mark.parametrize(
        "stats",
        [np.zeros(1), np.zeros(4), np.zeros((1, 2)), np.zeros((3, 2))],
        ids=["short_1d", "flat_1d", "too_few_groups", "too_many_groups"],
    )
    def test_rejects_stats_that_are_not_two_per_group(self, ga, call, stats):
        # every group reads stats[2 * i] and stats[2 * i + 1], so anything
        # shorter is read past its end
        with pytest.raises(ValueError, match=r"stats must have shape"):
            call(ga, stats)

    def test_accepts_two_stats_per_group(self, ga):
        stats = np.array([[0.0, 2.0], [0.0, 2.0]])
        np.testing.assert_allclose(ga._scaler_transform(stats), np.arange(10.0) / 2)


class TestSeasonLength:
    @pytest.mark.parametrize("season_length", [0, -3])
    def test_guerrero_rejects_a_non_positive_season_length(self, ga, season_length):
        # n_seasons = n / period, so zero divided by zero and a negative one
        # sized a vector with a huge value
        with pytest.raises(ValueError, match="season_length must be greater than 0"):
            boxcox_lambda(np.arange(1.0, 11.0), "guerrero", season_length)
        with pytest.raises(ValueError, match="season_length must be greater than 0"):
            LocalBoxCoxScaler("guerrero", season_length)
        with pytest.raises(ValueError, match="season_length must be greater than 0"):
            ga._boxcox_guerrero(season_length, -0.9, 2.0)

    @pytest.mark.parametrize(
        "call",
        [
            lambda ga, period: num_seas_diffs(np.arange(1.0, 11.0), period, 1),
            lambda ga, period: ga._num_seas_diffs(period, 1),
        ],
        ids=["free_fn", "grouped"],
    )
    def test_num_seas_diffs_rejects_a_negative_season_length(self, ga, call):
        with pytest.raises(ValueError, match="season_length must be non-negative"):
            call(ga, -2)

    def test_num_seas_diffs_accepts_zero(self, ga):
        # find_season_length passes the zero it gets when it finds no seasonality
        assert num_seas_diffs(np.arange(1.0, 11.0), 0, 1) == 0
        np.testing.assert_array_equal(ga._num_seas_diffs(0, 1), [0.0, 0.0])

    def test_find_season_length_without_seasonality(self):
        assert find_season_length(np.arange(1.0, 30.0), 10) == 0
