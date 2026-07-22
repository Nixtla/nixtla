"""Stochastic Volatility (Heston/SABR) time series generator."""

from typing import Literal

import numpy as np
from pydantic import Field

from synforecast._lib import volatility as _rs_vol
from synforecast.base import BaseGenerator


class StochasticVolatilityGenerator(BaseGenerator):
    """Generate time series where volatility itself follows a stochastic process.

    Heston model (variance is mean-reverting square-root/CIR):

        dS = mu * S dt + sqrt(V) * S dW1
        dV = kappa * (theta - V) dt + sigma_v * sqrt(V) dW2
        Corr(dW1, dW2) = rho

    SABR model (for rates/FX):

        dF = sigma * F^beta dW1
        dsigma = alpha * sigma dW2
        Corr(dW1, dW2) = rho

    Both are simulated with Euler-Maruyama; the Heston variance uses a
    truncation scheme (floored at a small positive value) so the discretized
    variance stays positive even when the Feller condition
    ``2 * kappa * theta > sigma_v^2`` is violated. Negative ``rho`` produces
    the leverage effect (volatility rises when prices fall).

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias (e.g. 'D', 'h', '5min') or an
            integer time step.
        model (str): 'heston' or 'sabr' (default: 'heston').
        initial_price (float): Starting price S0 (default: 100.0).
        initial_vol (float): Starting variance V0; SABR uses sqrt(initial_vol)
            as its starting volatility sigma0 (default: 0.04).
        drift (float): Price drift mu (default: 0.05).
        mean_vol (float): Long-run variance theta (Heston only, default: 0.04).
        vol_mean_reversion (float): Variance mean-reversion speed kappa
            (Heston only, default: 2.0).
        vol_of_vol (float): Volatility of volatility sigma_v (Heston) or
            alpha (SABR) (default: 0.3).
        correlation (float): Price-volatility correlation rho in [-1, 1]
            (default: -0.7).
        beta (float): CEV exponent in [0, 1] (SABR only; 0=normal,
            1=lognormal, default: 0.5).
        dt (float): Time step for discretization (default: 1/252).
        output_type (str): 'price', 'returns' (log returns), or 'volatility'
            (default: 'price').
        seed (int | None): Random seed for reproducibility (default: None).

    Example:
        >>> gen = StochasticVolatilityGenerator(
        ...     min_length=252,
        ...     max_length=252,
        ...     freq="D",
        ...     model="heston",
        ...     initial_price=100.0,
        ...     correlation=-0.7,  # Leverage effect
        ...     seed=42,
        ... )
        >>> df = gen.generate(n_series=10)
    """

    model: Literal["heston", "sabr"] = Field(
        default="heston", description="Stochastic volatility model"
    )
    initial_price: float = Field(default=100.0, gt=0.0, description="Starting price S0")
    initial_vol: float = Field(
        default=0.04,
        gt=0.0,
        description="Starting variance V0 (SABR starts at sqrt(initial_vol))",
    )
    drift: float = Field(default=0.05, description="Price drift mu")
    mean_vol: float = Field(
        default=0.04, gt=0.0, description="Long-run variance theta (Heston)"
    )
    vol_mean_reversion: float = Field(
        default=2.0, gt=0.0, description="Variance mean-reversion speed kappa"
    )
    vol_of_vol: float = Field(
        default=0.3,
        gt=0.0,
        description="Volatility of volatility sigma_v (Heston) or alpha (SABR)",
    )
    correlation: float = Field(
        default=-0.7,
        ge=-1.0,
        le=1.0,
        description="Price-volatility correlation rho (negative = leverage effect)",
    )
    beta: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="CEV exponent beta (SABR only, 0=normal, 1=lognormal)",
    )
    dt: float = Field(
        default=1 / 252,
        gt=0.0,
        description="Time step (1/252 = daily with 252 trading days)",
    )
    output_type: Literal["price", "returns", "volatility"] = Field(
        default="price", description="Output type: price, log returns, or volatility"
    )

    def _generate_correlated_brownians(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate correlated Brownian motion increments.

        Args:
            n: Number of increments

        Returns:
            tuple: (dW1, dW2) increments with Corr(dW1, dW2) = correlation
        """
        z1 = self._sample_innovations(n)
        z2 = self._sample_innovations(n)

        dW1 = z1 * np.sqrt(self.dt)
        dW2 = (self.correlation * z1 + np.sqrt(1 - self.correlation**2) * z2) * np.sqrt(
            self.dt
        )

        return dW1, dW2

    def _simulate_heston(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the Heston model via Euler-Maruyama.

        The variance is floored at a small positive value each step so the
        square-root diffusion is well-defined when discretization pushes it
        negative.

        Args:
            length: Number of time steps

        Returns:
            tuple: (prices, variances) arrays
        """
        prices = np.zeros(length)
        variances = np.zeros(length)

        prices[0] = self.initial_price
        variances[0] = self.initial_vol

        dW1, dW2 = self._generate_correlated_brownians(length - 1)

        for t in range(length - 1):
            S = prices[t]
            V = max(variances[t], 1e-8)

            # dS = mu * S dt + sqrt(V) * S dW1
            dS = self.drift * S * self.dt + np.sqrt(V) * S * dW1[t]
            prices[t + 1] = max(S + dS, 1e-8)

            # dV = kappa * (theta - V) dt + sigma_v * sqrt(V) dW2
            dV = (
                self.vol_mean_reversion * (self.mean_vol - V) * self.dt
                + self.vol_of_vol * np.sqrt(V) * dW2[t]
            )
            variances[t + 1] = max(V + dV, 1e-8)

        return prices, variances

    def _simulate_sabr(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the SABR model via Euler-Maruyama.

        dF = sigma * F^beta dW1
        dsigma = alpha * sigma dW2

        Args:
            length: Number of time steps

        Returns:
            tuple: (forwards, variances) arrays
        """
        forwards = np.zeros(length)
        vols = np.zeros(length)

        forwards[0] = self.initial_price
        vols[0] = np.sqrt(self.initial_vol)  # SABR evolves vol, not variance

        dW1, dW2 = self._generate_correlated_brownians(length - 1)

        for t in range(length - 1):
            F = max(forwards[t], 1e-8)
            sigma = max(vols[t], 1e-8)

            F_beta = F**self.beta if self.beta > 0 else 1.0
            dF = sigma * F_beta * dW1[t]
            forwards[t + 1] = max(F + dF, 1e-8)

            d_sigma = self.vol_of_vol * sigma * dW2[t]
            vols[t + 1] = max(sigma + d_sigma, 1e-8)

        return forwards, vols**2

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        model_t = 0 if self.model == "heston" else 1
        output_t = {"price": 0, "returns": 1, "volatility": 2}[self.output_type]
        return (
            np.array(
                [
                    float(model_t),
                    self.initial_price,
                    self.initial_vol,
                    self.drift,
                    self.mean_vol,
                    self.vol_mean_reversion,
                    self.vol_of_vol,
                    self.correlation,
                    self.beta,
                    self.dt,
                    float(output_t),
                    float(self._rs_innov_dist),
                    self._rs_innov_param,
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single stochastic volatility series.

        Args:
            length (int): The length of the series to generate

        Returns:
            np.ndarray: Array of values (price, returns, or volatility)
        """
        seed = int(self.rng.integers(0, 2**63))
        model_t = 0 if self.model == "heston" else 1
        output_t = {"price": 0, "returns": 1, "volatility": 2}[self.output_type]
        return _rs_vol.stochastic_volatility(
            length,
            model_t,
            self.initial_price,
            self.initial_vol,
            self.drift,
            self.mean_vol,
            self.vol_mean_reversion,
            self.vol_of_vol,
            self.correlation,
            self.beta,
            self.dt,
            output_t,
            seed,
            self._rs_innov_dist,
            self._rs_innov_param,
        )

    def generate_with_volatility(
        self, n_series: int = 1, start_id: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate series and return both prices and volatility paths.

        Args:
            n_series (int): Number of series to generate (default: 1)
            start_id (int): Starting ID for series naming (default: 0)

        Returns:
            tuple: (prices, volatilities, series_ids) arrays
        """
        all_prices = []
        all_vols = []
        all_ids = []

        for i in range(n_series):
            length = self.rng.integers(self.min_length, self.max_length + 1)

            if self.model == "heston":
                prices, variances = self._simulate_heston(length)
            else:
                prices, variances = self._simulate_sabr(length)

            all_prices.append(prices)
            all_vols.append(np.sqrt(variances))
            all_ids.append(np.full(length, start_id + i))

        return (
            np.concatenate(all_prices),
            np.concatenate(all_vols),
            np.concatenate(all_ids),
        )

    def get_model_info(self) -> dict:
        """Get information about the stochastic volatility model.

        Returns:
            dict: Model parameters and characteristics
        """
        info = {
            "model": self.model,
            "initial_price": self.initial_price,
            "initial_vol": self.initial_vol,
            "drift": self.drift,
            "vol_of_vol": self.vol_of_vol,
            "correlation": self.correlation,
            "leverage_effect": self.correlation < 0,
            "dt": self.dt,
            "output_type": self.output_type,
        }

        if self.model == "heston":
            info["mean_vol"] = self.mean_vol
            info["vol_mean_reversion"] = self.vol_mean_reversion
            # Feller condition 2*kappa*theta > sigma_v^2 keeps the continuous
            # process strictly positive
            feller = 2 * self.vol_mean_reversion * self.mean_vol - self.vol_of_vol**2
            info["feller_condition_satisfied"] = feller > 0
            info["feller_value"] = feller
        else:
            info["beta"] = self.beta
            beta_interpretation = {
                0.0: "normal model",
                0.5: "CIR-like",
                1.0: "lognormal model",
            }
            info["beta_interpretation"] = beta_interpretation.get(
                self.beta, f"CEV with beta={self.beta}"
            )

        return info

    def implied_volatility_smile(
        self, strikes: np.ndarray, maturity: float = 1.0
    ) -> np.ndarray:
        """Approximate implied volatility smile for given strikes.

        Uses the Hagan SABR approximation formula (valid for the SABR model;
        a rough approximation for Heston).

        Args:
            strikes: Array of strike prices
            maturity: Time to maturity in years

        Returns:
            Array of implied volatilities
        """
        F = self.initial_price
        sigma = np.sqrt(self.initial_vol)
        alpha = self.vol_of_vol
        rho = self.correlation
        beta = self.beta if self.model == "sabr" else 1.0

        impl_vols = np.zeros_like(strikes, dtype=float)

        for i, K in enumerate(strikes):
            if np.abs(F - K) < 1e-8:
                impl_vols[i] = sigma
            else:
                log_fk = np.log(F / K)
                FK_mid = np.sqrt(F * K)
                FK_beta = FK_mid ** (1 - beta)

                vol = sigma / FK_beta

                # Skew correction z / x(z)
                z = (alpha / sigma) * FK_beta * log_fk
                if np.abs(z) < 1e-8:
                    x_z = 1.0
                else:
                    x_z = z / np.log(
                        (np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho)
                    )

                vol = vol * x_z

                time_adj = (
                    1
                    + (
                        (1 - beta) ** 2 / 24 * sigma**2 / FK_beta**2
                        + rho * beta * alpha * sigma / (4 * FK_beta)
                        + (2 - 3 * rho**2) * alpha**2 / 24
                    )
                    * maturity
                )

                impl_vols[i] = vol * time_adj

        return impl_vols
