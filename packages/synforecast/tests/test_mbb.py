"""Check remainder block sampling independently of the native decomposition."""

import numpy as np
import pytest

from synforecast._lib import augmentation


@pytest.mark.parametrize("block", [2, 5, 7, 30, 31])
@pytest.mark.parametrize("batched", [False, True])
def test_sampled_remainders_form_requested_blocks(block: int, batched: bool) -> None:
    values = np.random.default_rng(315).normal(size=31)
    # For n=31 with no period the trend window is three, with edge extension.
    trend = np.pad(np.convolve(values, np.ones(3) / 3, "valid"), (1, 1), mode="edge")
    remainder = values - trend
    assert len(np.unique(remainder)) == len(values)
    seeds = list(range(64))
    copies = (
        augmentation.moving_block_bootstrap_many(values, block, seeds)
        if batched
        else [
            augmentation.moving_block_bootstrap(values, block, seed) for seed in seeds
        ]
    )
    observed_starts = set()
    for copy in copies:
        sampled = copy - trend
        # Unique remainders reveal the source indices without knowing the RNG.
        indices = np.abs(sampled[:, None] - remainder).argmin(axis=1)
        np.testing.assert_allclose(sampled, remainder[indices], atol=1e-12)
        possible_offsets = []
        for offset in range(block):
            positions = offset + np.arange(len(values))
            starts = indices - positions % block
            inside_block = np.diff(positions // block) == 0
            if np.all((starts >= 0) & (starts <= len(values) - block)) and np.all(
                np.diff(starts)[inside_block] == 0
            ):
                possible_offsets.append(offset)
                observed_starts.update(starts.tolist())
        assert possible_offsets, f"draw does not contain blocks of length {block}"
    # Both endpoints are eligible block starts (including the last one).
    assert {0, len(values) - block} <= observed_starts


def test_public_mbb_caps_block_per_source(monkeypatch) -> None:
    import polars as pl

    from synforecast import SynAugment

    original = augmentation.moving_block_bootstrap_many
    received = []

    def record(values, block_size, seeds, period=None):
        received.append((len(values), block_size, len(seeds)))
        return original(values, block_size, seeds, period)

    monkeypatch.setattr(augmentation, "moving_block_bootstrap_many", record)
    frame = pl.DataFrame(
        {
            "unique_id": ["a"] * 12 + ["b"] * 30,
            "ds": list(range(12)) + list(range(30)),
            "y": np.random.default_rng(10).normal(size=42),
        }
    )
    output = SynAugment(seed=4).mbb(
        frame, n_augment=3, block_size=7, seasonal_period=3, include_original=False
    )
    assert received == [(12, 4, 3), (30, 7, 3)]
    assert output.height == 3 * frame.height
