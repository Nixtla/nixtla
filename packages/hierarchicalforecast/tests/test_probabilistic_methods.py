import numpy as np
import pytest

from hierarchicalforecast.evaluation import (
    energy_score,
    log_score,
    msse,
    rel_mse,
    scaled_crps,
)
from hierarchicalforecast.methods import BottomUp
from hierarchicalforecast.probabilistic_methods import PERMBU, Bootstrap, Normality


@pytest.fixture
def test_data():
    """Fixture to provide test data for hierarchical forecasting tests."""
    S = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    h = 2
    _y = np.array([10.0, 5.0, 4.0, 2.0, 1.0])
    y_bottom = np.vstack([i * _y for i in range(1, 5)])
    y_hat_bottom_insample = np.roll(y_bottom, 1)
    y_hat_bottom_insample[:, 0] = np.nan
    y_hat_bottom = np.vstack([i * np.ones(h) for i in range(1, 5)])
    idx_bottom = [4, 3, 5, 6]
    tags = {"level1": np.array([0]), "level2": np.array([1, 2]), "level3": idx_bottom}

    # sigmah for all levels in the hierarchy
    # sigmah for Naive method
    # as calculated here:
    # https://otexts.com/fpp3/prediction-intervals.html
    y_base = S @ y_bottom
    y_hat_base = S @ y_hat_bottom
    y_hat_base_insample = S @ y_hat_bottom_insample
    sigma = np.nansum((y_base - y_hat_base_insample) ** 2, axis=1) / (y_base.shape[1] - 1)
    sigma = np.sqrt(sigma)
    sigmah = sigma[:, None] * np.sqrt(
        np.vstack([np.arange(1, h + 1) for _ in range(y_base.shape[0])])
    )
    noise = np.random.normal(scale=sigmah)
    y_test = y_hat_base + noise

    return {
        'S': S,
        'h': h,
        'y_bottom': y_bottom,
        'y_hat_bottom': y_hat_bottom,
        'y_hat_bottom_insample': y_hat_bottom_insample,
        'idx_bottom': idx_bottom,
        'tags': tags,
        'y_base': y_base,
        'y_hat_base': y_hat_base,
        'y_hat_base_insample': y_hat_base_insample,
        'sigmah': sigmah,
        'y_test': y_test,
    }


@pytest.fixture
def samplers(test_data):
    """Fixture to provide samplers for testing."""
    # samplers for tests
    cls_bottom_up = BottomUp()
    P, W = cls_bottom_up._get_PW_matrices(S=test_data['S'], idx_bottom=test_data['idx_bottom'])

    normality_sampler = Normality(
        S=test_data['S'],
        P=P,
        W=W,
        y_hat=test_data['y_hat_base'],
        sigmah=test_data['sigmah']
    )
    bootstrap_sampler = Bootstrap(
        S=test_data['S'],
        P=P,
        W=W,
        y_hat=test_data['y_hat_base'],
        y_insample=test_data['y_base'],
        y_hat_insample=test_data['y_hat_base_insample'],
        num_samples=1_000,
    )
    empty_bootstrap_sampler = Bootstrap(
        S=test_data['S'],
        P=P,
        W=W,
        y_hat=test_data['y_hat_base'],
        y_insample=test_data['y_base'],
        y_hat_insample=test_data['y_base'],
        num_samples=1_000,
    )
    permbu_sampler = PERMBU(
        S=test_data['S'],
        P=P,
        tags=test_data['tags'],
        y_hat=test_data['y_hat_base'],
        y_insample=test_data['y_base'],
        y_hat_insample=test_data['y_hat_base_insample'],
        sigmah=test_data['sigmah'],
    )
    empty_permbu_sampler = PERMBU(
        S=test_data['S'],
        P=P,
        tags=test_data['tags'],
        y_hat=test_data['y_hat_base'],
        y_insample=test_data['y_base'],
        y_hat_insample=test_data['y_base'],
        sigmah=test_data['sigmah'],
    )

    return {
        'normality_sampler': normality_sampler,
        'bootstrap_sampler': bootstrap_sampler,
        'empty_bootstrap_sampler': empty_bootstrap_sampler,
        'permbu_sampler': permbu_sampler,
        'empty_permbu_sampler': empty_permbu_sampler,
    }

def test_coherent_samples_shape(samplers):
    """Test that coherent samples have the correct shape."""
    normality_samples = samplers['normality_sampler'].get_samples(num_samples=100)
    bootstrap_samples = samplers['bootstrap_sampler'].get_samples(num_samples=100)
    permbu_samples = samplers['permbu_sampler'].get_samples(num_samples=100)

    assert bootstrap_samples.shape == normality_samples.shape
    assert bootstrap_samples.shape == permbu_samples.shape


def test_rel_mse_execution(test_data):
    """Test RelMSE's execution."""
    result = rel_mse(
        y=test_data['y_test'],
        y_hat=test_data['y_hat_base'],
        y_train=test_data['y_base']
    )
    assert result is not None


def test_msse_execution(test_data):
    """Test MSSE's execution."""
    result = msse(
        y=test_data['y_test'],
        y_hat=test_data['y_hat_base'],
        y_train=test_data['y_base']
    )
    assert result is not None


def test_energy_score_execution(test_data, samplers):
    """Test energy score's execution."""
    bootstrap_samples = samplers['bootstrap_sampler'].get_samples(num_samples=100)
    permbu_samples = samplers['permbu_sampler'].get_samples(num_samples=100)

    result = energy_score(
        y=test_data['y_test'],
        y_sample1=bootstrap_samples,
        y_sample2=permbu_samples
    )
    assert result is not None


def test_scaled_crps_execution(test_data, samplers):
    """Test scaled CRPS' execution."""
    bootstrap_samples = samplers['bootstrap_sampler'].get_samples(num_samples=100)
    quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    bootstrap_quantiles = np.quantile(bootstrap_samples, q=quantiles, axis=2)
    bootstrap_quantiles = bootstrap_quantiles.transpose((1, 2, 0))  # [Q,N,H] -> [N,H,Q]

    result = scaled_crps(
        y=test_data['y_test'],
        y_hat=bootstrap_quantiles,
        quantiles=quantiles
    )
    assert result is not None


def test_log_score_execution(test_data, samplers):
    """Test log score's execution."""
    cov = np.concatenate([cov[:, :, None] for cov in samplers['normality_sampler'].cov_rec], axis=2)
    result = log_score(
        y=test_data['y_test'],
        y_hat=test_data['y_hat_base'],
        cov=cov,
        allow_singular=True
    )
    assert result is not None


def test_quantile_loss_protections(test_data, samplers):
    """Test quantile loss protections."""
    bootstrap_samples = samplers['bootstrap_sampler'].get_samples(num_samples=100)
    quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    bootstrap_quantiles = np.quantile(bootstrap_samples, q=quantiles, axis=2)
    bootstrap_quantiles = bootstrap_quantiles.transpose((1, 2, 0))  # [Q,N,H] -> [N,H,Q]

    # Test with invalid quantiles (>1.0)
    invalid_quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.2])

    with pytest.raises(Exception) as exc_info:
        scaled_crps(test_data['y_test'], bootstrap_quantiles, invalid_quantiles)

    assert "between 0 and 1" in str(exc_info.value)
