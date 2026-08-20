"""Calculate portfolio weights used by the short-term factor model."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from bbg_cache import (
    SHORT_INTEREST_LOOKBACK_DAYS,
    TICKER_PATTERN,
    _read_workbook_cells,
    historical_price_windows,
)


Normalization = Literal["gross", "net"]
MAD_SCALE = 1.4826
CONVENTIONAL_Z = 3.0
FACTOR_ROBUST_Z = {
    "one_day_reversal": 6.0,
    "short_term_reversal": 5.0,
    "seasonality": 5.0,
    "industry_momentum": 5.0,
    "short_interest": 5.0,
    "downside_risk": 5.0,
    "size": 5.0,
}


def _ticker(value: str) -> str:
    """Return the base ticker from CSV or Bloomberg ticker text."""
    return value.strip().upper().split()[0]


def _warn_missing_data(
    label: str,
    values: pd.Series,
    required_tickers: pd.Index,
    preview_size: int = 20,
) -> None:
    """Warn about required tickers with no input value."""
    required = required_tickers.drop_duplicates()
    available = values[values.notna()].index
    missing = required[~required.isin(available)]
    if missing.empty:
        return

    preview = ", ".join(map(str, missing[:preview_size]))
    remainder = len(missing) - preview_size
    suffix = f", and {remainder:,} more" if remainder > 0 else ""
    print(
        f"WARNING: Missing {label} for {len(missing):,} of "
        f"{len(required):,} required tickers: {preview}{suffix}.",
        file=sys.stderr,
    )


def robust_standard_deviation(values: pd.Series) -> float:
    """Calculate 1.4826 times the median absolute deviation from the median."""
    median = values.median()
    return MAD_SCALE * (values - median).abs().median()


def winsorize_factor(
    values: pd.Series,
    factor: str | None = None,
    conventional_z: float = CONVENTIONAL_Z,
    estimation_index: pd.Index | None = None,
) -> pd.Series:
    """Winsorize values using robust and then conventional deviation limits."""
    factor = factor or values.name
    robust_z = FACTOR_ROBUST_Z[factor]
    reference = values if estimation_index is None else values.reindex(estimation_index)
    median = reference.median()
    robust_std = robust_standard_deviation(reference)
    trimmed = values.clip(
        median - robust_z * robust_std,
        median + robust_z * robust_std,
    )

    trimmed_reference = reference.clip(
        median - robust_z * robust_std,
        median + robust_z * robust_std,
    )
    mean = trimmed_reference.mean()
    standard_deviation = trimmed_reference.std()
    return trimmed.clip(
        mean - conventional_z * standard_deviation,
        mean + conventional_z * standard_deviation,
    ).rename(values.name)


def normalize_factor_exposures(
    trimmed_values: pd.Series,
    market_caps: pd.Series,
    estimation_index: pd.Index | None = None,
) -> pd.Series:
    """Standardize a descriptor and make its cap-weighted exposure zero."""
    reference = (
        trimmed_values
        if estimation_index is None
        else trimmed_values.reindex(estimation_index)
    )
    standardized = (
        trimmed_values - reference.mean()
    ) / reference.std(ddof=0)
    reference_exposures = standardized.reindex(reference.index)
    reference_caps = market_caps.reindex(reference.index)
    valid = reference_exposures.notna() & reference_caps.notna()
    cap_weighted_mean = (
        reference_exposures[valid] * reference_caps[valid]
    ).sum() / reference_caps[valid].sum()
    return (standardized - cap_weighted_mean).rename(trimmed_values.name)


def load_benchmark_tickers(xlsx_path: str | Path) -> pd.Index:
    """Return tickers from the Ticker/Bmrk column of an estimation workbook."""
    cells = _read_workbook_cells(Path(xlsx_path))
    sheet, header_row, column = next(
        (sheet, row, column)
        for (sheet, row, column), value in cells.items()
        if value.casefold() == "bmrk"
    )
    last_row = max(row for cell_sheet, row, _ in cells if cell_sheet == sheet)
    tickers = dict.fromkeys(
        cells.get((sheet, row, column), "").strip().upper()
        for row in range(header_row + 1, last_row + 1)
        if TICKER_PATTERN.fullmatch(cells.get((sheet, row, column), "").strip())
    )
    return pd.Index(tickers, name="ticker")


def load_portfolio_positions(
    csv_path: str | Path,
) -> tuple[date, pd.Series]:
    """Return the portfolio date and shares by ticker from a position CSV."""
    positions = pd.read_csv(csv_path)
    as_of_date = datetime.strptime(positions["Date"].iloc[0], "%m/%d/%Y").date()
    positions["ticker"] = positions["ticker"].map(_ticker)
    shares = positions.set_index("ticker")["shares"].astype(float)
    shares.name = "shares"
    return as_of_date, shares


def load_latest_prices(
    database_path: str | Path,
    tickers: pd.Index,
    as_of_date: date,
) -> pd.Series:
    """Return each ticker's latest closing price through the portfolio date."""
    query = """
        SELECT ticker, price
        FROM prices
        WHERE date <= ?
        ORDER BY date DESC
    """
    connection = sqlite3.connect(database_path)
    prices = pd.read_sql_query(
        query,
        connection,
        params=(as_of_date.isoformat(),),
    )
    connection.close()
    prices["ticker"] = prices["ticker"].map(_ticker)
    prices = prices[prices["ticker"].isin(tickers)].drop_duplicates("ticker")
    return prices.set_index("ticker")["price"].astype(float)


def load_latest_market_caps(
    database_path: str | Path,
    tickers: pd.Index,
    as_of_date: date,
) -> pd.Series:
    """Return each ticker's market cap on the rebalance date."""
    query = """
        SELECT ticker, market_cap
        FROM market_caps
        WHERE date = ?
    """
    connection = sqlite3.connect(database_path)
    market_caps = pd.read_sql_query(
        query,
        connection,
        params=(as_of_date.isoformat(),),
    )
    connection.close()
    market_caps["ticker"] = market_caps["ticker"].map(_ticker)
    market_caps = market_caps[
        market_caps["ticker"].isin(tickers)
    ].drop_duplicates("ticker")
    return market_caps.set_index("ticker")["market_cap"].astype(float)


def load_latest_short_interest(
    database_path: str | Path,
    tickers: pd.Index,
    as_of_date: date,
) -> pd.Series:
    """Return the latest short interest within 21 calendar days."""
    start_date = as_of_date - timedelta(days=SHORT_INTEREST_LOOKBACK_DAYS)
    query = """
        SELECT ticker, short_interest_percent_float
        FROM short_interest
        WHERE date BETWEEN ? AND ?
        ORDER BY date DESC
    """
    connection = sqlite3.connect(database_path)
    short_interest = pd.read_sql_query(
        query,
        connection,
        params=(start_date.isoformat(), as_of_date.isoformat()),
    )
    connection.close()
    short_interest["ticker"] = short_interest["ticker"].map(_ticker)
    short_interest = short_interest[
        short_interest["ticker"].isin(tickers)
    ].drop_duplicates("ticker")
    return short_interest.set_index("ticker")[
        "short_interest_percent_float"
    ].astype(float)


def get_size(market_caps: pd.Series) -> pd.Series:
    """Return the natural logarithm of market capitalization."""
    return np.log(market_caps).rename("size")


def calculate_portfolio_weights(
    shares: pd.Series,
    prices: pd.Series,
    normalization: Normalization = "gross",
) -> pd.Series:
    """Calculate signed market-value weights by ticker."""
    market_values = shares.mul(prices).dropna()
    denominator = (
        market_values.abs().sum()
        if normalization == "gross"
        else market_values.sum()
    )
    return (market_values / denominator).rename("weight")


def get_portfolio_weights(
    csv_path: str | Path,
    database_path: str | Path = "bloomberg_history.db",
    normalization: Normalization = "gross",
) -> pd.Series:
    """Read a position CSV and return weights using cached Bloomberg prices."""
    as_of_date, shares = load_portfolio_positions(csv_path)
    prices = load_latest_prices(database_path, shares.index, as_of_date)
    return calculate_portfolio_weights(shares, prices, normalization)


def get_one_day_reversal(
    weights: pd.Series,
    as_of_date: date,
    database_path: str | Path = "bloomberg_history.db",
) -> pd.Series:
    """Return the negative return known before the last-hour rebalance."""
    query = """
        SELECT ticker, date, price
        FROM prices
        WHERE date <= ?
        ORDER BY date
    """
    connection = sqlite3.connect(database_path)
    prices = pd.read_sql_query(
        query,
        connection,
        params=(as_of_date.isoformat(),),
    )
    connection.close()
    prices["ticker"] = prices["ticker"].map(_ticker)
    prices = prices[prices["ticker"].isin(weights.index)]
    prices = prices.pivot(index="date", columns="ticker", values="price")
    reversal = -prices.pct_change(fill_method=None).iloc[-2]
    return reversal.reindex(weights.index).rename("one_day_reversal")


def get_short_term_reversal(
    weights: pd.Series,
    as_of_date: date,
    database_path: str | Path = "bloomberg_history.db",
    half_life: int = 10,
) -> pd.Series:
    """Return short term reversal from returns at lags 2 through 21."""
    query = """
        SELECT ticker, date, price
        FROM prices
        WHERE date <= ?
        ORDER BY date
    """
    connection = sqlite3.connect(database_path)
    prices = pd.read_sql_query(
        query,
        connection,
        params=(as_of_date.isoformat(),),
    )
    connection.close()
    prices["ticker"] = prices["ticker"].map(_ticker)
    prices = prices[prices["ticker"].isin(weights.index)]
    prices = prices.pivot(index="date", columns="ticker", values="price")

    returns = prices.pct_change(fill_method=None).iloc[-21:-1].iloc[::-1]
    decay = pd.Series(0.5 ** (pd.RangeIndex(20) / half_life))
    reversal = -returns.mul(decay.to_numpy(), axis=0).sum() / decay.sum()
    return reversal.reindex(weights.index).rename("short_term_reversal")


def get_downside_risk(
    weights: pd.Series,
    as_of_date: date,
    database_path: str | Path = "bloomberg_history.db",
    lookback_days: int = 126,
    target_return: float = 0.0,
) -> pd.Series:
    """Return downside variance over returns known before the rebalance."""
    query = """
        SELECT ticker, date, price
        FROM prices
        WHERE date <= ?
        ORDER BY date
    """
    connection = sqlite3.connect(database_path)
    prices = pd.read_sql_query(
        query,
        connection,
        params=(as_of_date.isoformat(),),
    )
    connection.close()
    prices["ticker"] = prices["ticker"].map(_ticker)
    prices = prices[prices["ticker"].isin(weights.index)]
    prices = prices.pivot(index="date", columns="ticker", values="price")

    returns = prices.pct_change(fill_method=None).iloc[
        -(lookback_days + 1) : -1
    ]
    downside_variance = returns.sub(target_return).clip(upper=0).pow(2).mean()
    return downside_variance.reindex(weights.index).rename("downside_risk")


def get_seasonality(
    weights: pd.Series,
    as_of_date: date,
    database_path: str | Path = "bloomberg_history.db",
) -> pd.Series:
    """Return the mean return from the same one-month window over five years."""
    window_returns = []
    connection = sqlite3.connect(database_path)
    try:
        for start_date, end_date in historical_price_windows(as_of_date)[1:]:
            prices = pd.read_sql_query(
                """
                SELECT ticker, date, price
                FROM prices
                WHERE date BETWEEN ? AND ?
                ORDER BY date
                """,
                connection,
                params=(start_date.isoformat(), end_date.isoformat()),
            )
            prices["ticker"] = prices["ticker"].map(_ticker)
            prices = prices[prices["ticker"].isin(weights.index)]
            grouped_prices = prices.groupby("ticker")["price"]
            window_returns.append(
                grouped_prices.last().div(grouped_prices.first()).sub(1.0)
            )
    finally:
        connection.close()

    seasonality = pd.concat(window_returns, axis=1).mean(axis=1)
    return seasonality.reindex(weights.index).rename("seasonality")


def get_industry_momentum(
    weights: pd.Series,
    market_caps: pd.Series,
    as_of_date: date,
    database_path: str | Path = "bloomberg_history.db",
    lookback_days: int = 180,
) -> pd.Series:
    """Return each stock's cap-weighted BICS industry momentum exposure."""
    query = """
        SELECT ticker, date, industry, price
        FROM prices
        WHERE date <= ?
        ORDER BY date
    """
    connection = sqlite3.connect(database_path)
    prices = pd.read_sql_query(
        query,
        connection,
        params=(as_of_date.isoformat(),),
    )
    connection.close()
    prices["ticker"] = prices["ticker"].map(_ticker)
    prices = prices[prices["ticker"].isin(weights.index)]

    industries = (
        prices.drop_duplicates("ticker", keep="last")
        .set_index("ticker")["industry"]
    )
    price_history = prices.pivot(index="date", columns="ticker", values="price")
    return_prices = price_history.iloc[-(lookback_days + 2) : -1]
    first_prices = return_prices.bfill().iloc[0]
    last_prices = return_prices.ffill().iloc[-1]
    stock_returns = last_prices.div(first_prices).sub(1.0)
    stock_returns[return_prices.count() < 2] = pd.NA

    reference_returns = stock_returns.reindex(market_caps.index)
    reference_industries = industries.reindex(market_caps.index)
    reference_caps = market_caps.reindex(market_caps.index)
    valid = (
        reference_returns.notna()
        & reference_industries.notna()
        & reference_caps.notna()
    )
    weighted_returns = reference_returns[valid] * reference_caps[valid]
    industry_returns = weighted_returns.groupby(
        reference_industries[valid]
    ).sum() / reference_caps[valid].groupby(reference_industries[valid]).sum()

    momentum = industries.reindex(weights.index).map(industry_returns)
    return momentum.rename("industry_momentum")


def calculate_raw_factor_values(
    tickers: pd.Index,
    estimation_index: pd.Index,
    rebalance_date: date,
    database_path: str | Path = "bloomberg_history.db",
) -> tuple[dict[str, pd.Series], pd.Series]:
    """Calculate raw factor descriptors and market caps for a stock universe."""
    ticker_selector = pd.Series(index=tickers, dtype=float)
    market_caps = load_latest_market_caps(
        database_path,
        tickers,
        rebalance_date,
    )
    raw_factors = {
        "one_day_reversal": get_one_day_reversal(
            ticker_selector,
            rebalance_date,
            database_path,
        ),
        "short_term_reversal": get_short_term_reversal(
            ticker_selector,
            rebalance_date,
            database_path,
        ),
        "seasonality": get_seasonality(
            ticker_selector,
            rebalance_date,
            database_path,
        ),
        "industry_momentum": get_industry_momentum(
            ticker_selector,
            market_caps.reindex(estimation_index),
            rebalance_date,
            database_path,
        ),
        "short_interest": load_latest_short_interest(
            database_path,
            tickers,
            rebalance_date,
        ).reindex(tickers),
        "downside_risk": get_downside_risk(
            ticker_selector,
            rebalance_date,
            database_path,
        ),
        "size": get_size(market_caps).reindex(tickers),
    }
    return raw_factors, market_caps


def calculate_normalized_factor_exposures(
    tickers: pd.Index,
    estimation_index: pd.Index,
    rebalance_date: date,
    database_path: str | Path = "bloomberg_history.db",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return normalized stock-level exposures and corresponding market caps."""
    raw_factors, market_caps = calculate_raw_factor_values(
        tickers,
        estimation_index,
        rebalance_date,
        database_path,
    )
    benchmark_caps = market_caps.reindex(estimation_index)
    exposures = {}
    for factor, raw_values in raw_factors.items():
        trimmed = winsorize_factor(
            raw_values,
            factor=factor,
            estimation_index=estimation_index,
        )
        exposures[factor] = normalize_factor_exposures(
            trimmed,
            benchmark_caps,
            estimation_index=estimation_index,
        )
    return pd.DataFrame(exposures).reindex(tickers), market_caps


def calculate_factor_exposure_summary(
    rebalance_date: date,
    port_data_dir: str | Path = "port_data",
    database_path: str | Path = "bloomberg_history.db",
    portfolio_prefix: str = "US_live_port",
) -> pd.DataFrame:
    """Calculate portfolio and benchmark exposures for all factors."""
    date_stamp = rebalance_date.strftime("%Y%m%d")
    data_dir = Path(port_data_dir)
    portfolio_path = data_dir / f"{portfolio_prefix}_{date_stamp}.csv"
    universe_path = data_dir / f"estimation_universe_{date_stamp}.xlsx"

    _, shares = load_portfolio_positions(portfolio_path)
    benchmark_tickers = load_benchmark_tickers(universe_path)
    all_tickers = benchmark_tickers.append(shares.index).drop_duplicates()

    prices = load_latest_prices(database_path, shares.index, rebalance_date)
    _warn_missing_data("portfolio prices", prices, shares.index)
    portfolio_weights = calculate_portfolio_weights(shares, prices)
    print(f"Portfolio price coverage: {len(prices)}/{len(shares)}")
    raw_factors, market_caps = calculate_raw_factor_values(
        all_tickers,
        benchmark_tickers,
        rebalance_date,
        database_path,
    )
    _warn_missing_data(
        "benchmark market caps",
        market_caps,
        benchmark_tickers,
    )

    rows = []
    for factor, raw_values in raw_factors.items():
        _warn_missing_data(
            f"{factor} benchmark data",
            raw_values,
            benchmark_tickers,
        )
        _warn_missing_data(
            f"{factor} portfolio data",
            raw_values,
            portfolio_weights.index,
        )
        trimmed = winsorize_factor(
            raw_values,
            factor=factor,
            estimation_index=benchmark_tickers,
        )
        exposures = normalize_factor_exposures(
            trimmed,
            market_caps.reindex(benchmark_tickers),
            estimation_index=benchmark_tickers,
        )
        benchmark_exposures = exposures.reindex(benchmark_tickers)
        benchmark_caps = market_caps.reindex(benchmark_tickers)
        valid = benchmark_exposures.notna() & benchmark_caps.notna()
        rows.append(
            {
                "factor": factor,
                "portfolio_exposure": (
                    portfolio_weights * exposures.reindex(portfolio_weights.index)
                ).sum(),
                "benchmark_exposure": benchmark_exposures[valid].dot(
                    benchmark_caps[valid]
                )
                / benchmark_caps[valid].sum(),
            }
        )
    return pd.DataFrame(rows).set_index("factor")


def get_previous_price_date(
    database_path: str | Path,
    return_date: date,
) -> date:
    """Return the latest stored price date before the requested return date."""
    connection = sqlite3.connect(database_path)
    try:
        row = connection.execute(
            "SELECT MAX(date) FROM prices WHERE date < ?",
            (return_date.isoformat(),),
        ).fetchone()
    finally:
        connection.close()

    if row is None or row[0] is None:
        raise ValueError(f"No price date exists before {return_date}.")
    return date.fromisoformat(row[0])


def main(arguments: Sequence[str] | None = None) -> int:
    """Run factor exposures for a YYYYMMDD return date."""
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="Return date in YYYYMMDD format.")
    parser.add_argument(
        "--prefix",
        default="US_live_port",
        help=(
            "Portfolio filename prefix before _YYYYMMDD.csv "
            "(default: US_live_port)."
        ),
    )
    options = parser.parse_args(arguments)
    return_date = datetime.strptime(options.date, "%Y%m%d").date()
    exposure_date = get_previous_price_date(
        "bloomberg_history.db",
        return_date,
    )

    exposures = calculate_factor_exposure_summary(
        exposure_date,
        portfolio_prefix=options.prefix,
    )
    print(f"Exposure date: {exposure_date}; return date: {return_date}.")
    print(exposures.to_string(float_format=lambda value: f"{value:.8f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
