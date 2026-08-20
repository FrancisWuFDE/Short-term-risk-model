"""Forecast constant-maturity SPY volatility from current option prices."""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

TARGET_HORIZONS = {
    "one_day": 1,
    "one_month": 30,
    "one_year": 365,
}


def black_scholes_price(
    option_type: str,
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """Return a European option price under Black-Scholes-Merton."""
    volatility_time = volatility * np.sqrt(maturity)
    d1 = (
        np.log(spot / strike)
        + (
            risk_free_rate
            - dividend_yield
            + volatility**2 / 2.0
        )
        * maturity
    ) / volatility_time
    d2 = d1 - volatility_time
    discount_rate = np.exp(-risk_free_rate * maturity)
    discount_dividend = np.exp(-dividend_yield * maturity)

    if option_type == "call":
        return (
            spot * discount_dividend * norm.cdf(d1)
            - strike * discount_rate * norm.cdf(d2)
        )
    return (
        strike * discount_rate * norm.cdf(-d2)
        - spot * discount_dividend * norm.cdf(-d1)
    )


def implied_volatility(
    option_type: str,
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> float:
    """Invert an option midpoint to obtain implied volatility."""
    def pricing_error(volatility: float) -> float:
        return black_scholes_price(
            option_type,
            spot,
            strike,
            maturity,
            risk_free_rate,
            dividend_yield,
            volatility,
        ) - market_price

    return brentq(pricing_error, 0.0001, 5.0)


def _single_column(data: pd.DataFrame, column: str) -> pd.Series:
    """Return one yfinance field as a Series across column layouts."""
    values = data[column]
    return values.iloc[:, 0] if isinstance(values, pd.DataFrame) else values


def load_market_inputs(
    ticker: yf.Ticker,
    risk_free_rate: float | None = None,
) -> tuple[float, float, float]:
    """Return spot, approximate risk-free rate, and trailing dividend yield."""
    history = ticker.history(
        period="1y",
        auto_adjust=False,
        actions=True,
    )
    if history.empty:
        raise ValueError("No underlying price history was returned.")
    spot = float(history["Close"].dropna().iloc[-1])
    dividend_yield = float(history.get("Dividends", pd.Series(dtype=float)).sum())
    dividend_yield /= spot

    if risk_free_rate is None:
        treasury = yf.download(
            "^IRX",
            period="5d",
            auto_adjust=False,
            progress=False,
        )
        if treasury.empty:
            raise ValueError("No ^IRX risk-free-rate proxy was returned.")
        risk_free_rate = float(_single_column(treasury, "Close").dropna().iloc[-1])
        risk_free_rate /= 100.0
    return spot, risk_free_rate, dividend_yield


def select_expirations(
    expirations: Sequence[str],
    valuation_date: date,
) -> tuple[list[str], dict[str, tuple[str, str]]]:
    """Select expirations bracketing each constant-maturity target."""
    dated_expirations = sorted(
        (
            (datetime.strptime(expiration, "%Y-%m-%d").date(), expiration)
            for expiration in expirations
        ),
        key=lambda item: item[0],
    )
    dated_expirations = [
        (expiration_date, expiration)
        for expiration_date, expiration in dated_expirations
        if expiration_date > valuation_date
    ]
    if not dated_expirations:
        raise ValueError("No future option expirations were returned.")

    selected = set()
    brackets = {}
    for horizon, target_days in TARGET_HORIZONS.items():
        lower = [
            item
            for item in dated_expirations
            if (item[0] - valuation_date).days <= target_days
        ]
        upper = [
            item
            for item in dated_expirations
            if (item[0] - valuation_date).days >= target_days
        ]
        lower_expiration = lower[-1] if lower else dated_expirations[0]
        upper_expiration = upper[0] if upper else dated_expirations[-1]
        brackets[horizon] = (
            lower_expiration[1],
            upper_expiration[1],
        )
        selected.update(brackets[horizon])
    return sorted(selected), brackets


def calculate_expiration_volatility(
    options: pd.DataFrame,
    option_type: str,
    spot: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    options_per_side: int,
    max_relative_spread: float,
) -> pd.DataFrame:
    """Calculate midpoint IV for liquid options nearest the money."""
    options = options.copy()
    options["option_type"] = option_type
    options["midpoint"] = (options["bid"] + options["ask"]) / 2.0
    options["relative_spread"] = (
        options["ask"] - options["bid"]
    ) / options["midpoint"]
    options["absolute_log_moneyness"] = np.abs(
        np.log(options["strike"] / spot)
    )
    valid = (
        options["bid"].gt(0)
        & options["ask"].ge(options["bid"])
        & options["midpoint"].gt(0)
        & options["relative_spread"].le(max_relative_spread)
    )
    options = options[valid].nsmallest(
        options_per_side,
        "absolute_log_moneyness",
    )

    calculated_volatility = []
    for row in options.itertuples():
        try:
            volatility = implied_volatility(
                option_type,
                float(row.midpoint),
                spot,
                float(row.strike),
                maturity,
                risk_free_rate,
                dividend_yield,
            )
        except ValueError:
            volatility = np.nan
        calculated_volatility.append(volatility)
    options["calculated_implied_volatility"] = calculated_volatility
    return options.dropna(subset=["calculated_implied_volatility"])


def download_expiration_volatilities(
    symbol: str = "SPY",
    risk_free_rate: float | None = None,
    options_per_side: int = 3,
    max_relative_spread: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[str, str]]]:
    """Download selected option chains and estimate ATM IV by expiration."""
    valuation_date = date.today()
    ticker = yf.Ticker(symbol)
    spot, risk_free_rate, dividend_yield = load_market_inputs(
        ticker,
        risk_free_rate,
    )
    expirations, brackets = select_expirations(
        ticker.options,
        valuation_date,
    )

    option_rows = []
    expiration_rows = []
    for expiration in expirations:
        expiration_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        days_to_expiration = (expiration_date - valuation_date).days
        maturity = days_to_expiration / 365.0
        chain = ticker.option_chain(expiration)
        calls = calculate_expiration_volatility(
            chain.calls,
            "call",
            spot,
            maturity,
            risk_free_rate,
            dividend_yield,
            options_per_side,
            max_relative_spread,
        )
        puts = calculate_expiration_volatility(
            chain.puts,
            "put",
            spot,
            maturity,
            risk_free_rate,
            dividend_yield,
            options_per_side,
            max_relative_spread,
        )
        selected_options = pd.concat([calls, puts], ignore_index=True)
        if selected_options.empty:
            print(
                f"WARNING: No usable options for {expiration}.",
                file=sys.stderr,
            )
            continue
        selected_options["expiration"] = expiration
        selected_options["days_to_expiration"] = days_to_expiration
        selected_options["spot"] = spot
        selected_options["risk_free_rate"] = risk_free_rate
        selected_options["dividend_yield"] = dividend_yield
        option_rows.append(selected_options)
        expiration_rows.append(
            {
                "expiration": expiration,
                "days_to_expiration": days_to_expiration,
                "annualized_implied_volatility": selected_options[
                    "calculated_implied_volatility"
                ].median(),
                "options_used": len(selected_options),
            }
        )

    if not expiration_rows:
        raise ValueError("No expiration produced a usable implied volatility.")
    expiration_volatilities = pd.DataFrame(expiration_rows).set_index(
        "expiration"
    )
    selected_chain = pd.concat(option_rows, ignore_index=True)
    print(
        f"{symbol} spot: {spot:.2f}; risk-free rate: {risk_free_rate:.4%}; "
        f"dividend yield: {dividend_yield:.4%}."
    )
    return expiration_volatilities, selected_chain, brackets


def interpolate_constant_maturity(
    expiration_volatilities: pd.DataFrame,
    lower_expiration: str,
    upper_expiration: str,
    target_days: int,
) -> tuple[float, str]:
    """Interpolate total variance to a requested constant maturity."""
    available = expiration_volatilities
    requested = [lower_expiration, upper_expiration]
    usable = [expiration for expiration in requested if expiration in available.index]
    if not usable:
        nearest_index = (
            available["days_to_expiration"] - target_days
        ).abs().idxmin()
        usable = [nearest_index]
    if len(set(usable)) == 1:
        volatility = float(
            available.loc[usable[0], "annualized_implied_volatility"]
        )
        return volatility, "nearest_expiration"

    lower = available.loc[usable[0]]
    upper = available.loc[usable[1]]
    lower_days = float(lower["days_to_expiration"])
    upper_days = float(upper["days_to_expiration"])
    lower_variance = (
        float(lower["annualized_implied_volatility"]) ** 2
        * lower_days
        / 365.0
    )
    upper_variance = (
        float(upper["annualized_implied_volatility"]) ** 2
        * upper_days
        / 365.0
    )
    weight = (target_days - lower_days) / (upper_days - lower_days)
    target_variance = lower_variance + weight * (
        upper_variance - lower_variance
    )
    volatility = np.sqrt(target_variance / (target_days / 365.0))
    return float(volatility), "total_variance_interpolation"


def forecast_option_implied_volatility(
    symbol: str = "SPY",
    risk_free_rate: float | None = None,
    options_per_side: int = 3,
    max_relative_spread: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return constant-maturity option-implied volatility forecasts."""
    expiration_volatilities, selected_chain, brackets = (
        download_expiration_volatilities(
            symbol,
            risk_free_rate,
            options_per_side,
            max_relative_spread,
        )
    )
    forecasts = []
    for horizon, target_days in TARGET_HORIZONS.items():
        lower_expiration, upper_expiration = brackets[horizon]
        volatility, method = interpolate_constant_maturity(
            expiration_volatilities,
            lower_expiration,
            upper_expiration,
            target_days,
        )
        forecasts.append(
            {
                "horizon": horizon,
                "target_days": target_days,
                "annualized_implied_volatility_pct": volatility * 100.0,
                "expected_move_pct": (
                    volatility * np.sqrt(target_days / 365.0) * 100.0
                ),
                "lower_expiration": lower_expiration,
                "upper_expiration": upper_expiration,
                "method": method,
            }
        )
    return (
        pd.DataFrame(forecasts).set_index("horizon"),
        expiration_volatilities,
        selected_chain,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Download SPY options and print implied-volatility forecasts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol.")
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        help="Annual decimal rate, such as 0.04; defaults to ^IRX.",
    )
    parser.add_argument(
        "--options-per-side",
        type=int,
        default=3,
        help="Nearest strikes per call/put side and expiration.",
    )
    parser.add_argument(
        "--max-relative-spread",
        type=float,
        default=0.50,
        help="Maximum bid/ask spread divided by midpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV destination for constant-maturity forecasts.",
    )
    parser.add_argument(
        "--options-output",
        type=Path,
        help="Optional CSV destination for selected option prices and IVs.",
    )
    options = parser.parse_args(arguments)
    if options.options_per_side <= 0:
        parser.error("--options-per-side must be positive.")
    if options.max_relative_spread <= 0:
        parser.error("--max-relative-spread must be positive.")

    try:
        forecasts, expiration_volatilities, selected_chain = (
            forecast_option_implied_volatility(
                options.symbol,
                options.risk_free_rate,
                options.options_per_side,
                options.max_relative_spread,
            )
        )
    except (KeyError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("\nExpiration-level ATM implied volatility:")
    print(
        expiration_volatilities.to_string(
            float_format=lambda value: f"{value:.8f}"
        )
    )
    print("\nConstant-maturity volatility forecasts:")
    print(forecasts.to_string(float_format=lambda value: f"{value:.4f}"))

    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        forecasts.reset_index().to_csv(options.output, index=False)
        print(f"Saved forecasts to {options.output}.")
    if options.options_output:
        options.options_output.parent.mkdir(parents=True, exist_ok=True)
        selected_chain.to_csv(options.options_output, index=False)
        print(f"Saved selected options to {options.options_output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
