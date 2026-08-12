"""Calculate portfolio weights used by the short-term factor model."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd


Normalization = Literal["gross", "net"]


def _ticker(value: str) -> str:
    """Return the base ticker from CSV or Bloomberg ticker text."""
    return value.strip().upper().split()[0]


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


def calculate_portfolio_weights(
    shares: pd.Series,
    prices: pd.Series,
    normalization: Normalization = "gross",
) -> pd.Series:
    """Calculate signed market-value weights by ticker."""
    market_values = shares * prices.loc[shares.index]
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
    """Return each ticker's negative latest one-day return."""
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
    reversal = -prices.pct_change(fill_method=None).iloc[-1]
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
