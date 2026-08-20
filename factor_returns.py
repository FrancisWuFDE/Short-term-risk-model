"""Estimate next-day factor returns with a cross-sectional regression."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from short_term_factors import (
    _ticker,
    calculate_normalized_factor_exposures,
    get_previous_price_date,
    load_benchmark_tickers,
)


@dataclass(frozen=True)
class FactorReturnResult:
    """Results from one daily cross-sectional factor regression."""

    exposure_date: date
    return_date: date
    factor_returns: pd.Series
    observations: int
    r_squared: float
    rank: int


def load_stock_returns(
    database_path: str | Path,
    tickers: pd.Index,
    exposure_date: date,
    return_date: date,
) -> pd.Series:
    """Return close-to-close returns for two specified price dates."""
    connection = sqlite3.connect(database_path)
    return_date_row = connection.execute(
        "SELECT 1 FROM prices WHERE date = ? LIMIT 1",
        (return_date.isoformat(),),
    ).fetchone()
    if return_date_row is None:
        connection.close()
        raise ValueError(f"No prices exist on return date {return_date}.")

    prices = pd.read_sql_query(
        """
        SELECT ticker, date, price
        FROM prices
        WHERE date IN (?, ?)
        """,
        connection,
        params=(exposure_date.isoformat(), return_date.isoformat()),
    )
    connection.close()
    prices["ticker"] = prices["ticker"].map(_ticker)
    prices = prices[prices["ticker"].isin(tickers)]
    prices = prices.pivot_table(
        index="ticker",
        columns="date",
        values="price",
        aggfunc="last",
    )
    if exposure_date.isoformat() not in prices or return_date.isoformat() not in prices:
        raise ValueError(
            f"Prices are required on both {exposure_date} and {return_date}."
        )

    returns = prices[return_date.isoformat()].div(
        prices[exposure_date.isoformat()]
    ).sub(1.0)
    return returns.reindex(tickers).rename("stock_return")


def _warn_dropped_tickers(
    tickers: pd.Index,
    valid: pd.Series,
    preview_size: int = 20,
) -> None:
    """Warn about securities excluded from the regression."""
    missing = tickers[~valid.reindex(tickers, fill_value=False)]
    if missing.empty:
        return

    preview = ", ".join(map(str, missing[:preview_size]))
    remainder = len(missing) - preview_size
    suffix = f", and {remainder:,} more" if remainder > 0 else ""
    print(
        f"WARNING: Dropped {len(missing):,} of {len(tickers):,} securities "
        f"with incomplete regression data: {preview}{suffix}.",
        file=sys.stderr,
    )


def estimate_factor_returns(
    return_date: date,
    port_data_dir: str | Path = "port_data",
    database_path: str | Path = "bloomberg_history.db",
) -> FactorReturnResult:
    """Estimate factor returns for one specified return date."""
    exposure_date = get_previous_price_date(database_path, return_date)
    date_stamp = exposure_date.strftime("%Y%m%d")
    universe_path = Path(port_data_dir) / f"estimation_universe_{date_stamp}.xlsx"
    tickers = load_benchmark_tickers(universe_path)
    exposures, market_caps = calculate_normalized_factor_exposures(
        tickers,
        tickers,
        exposure_date,
        database_path,
    )
    stock_returns = load_stock_returns(
        database_path,
        tickers,
        exposure_date,
        return_date,
    )

    design = exposures.replace([np.inf, -np.inf], np.nan)
    design.insert(0, "market", 1.0)
    regression_weights = np.sqrt(market_caps.reindex(tickers))
    valid = (
        stock_returns.replace([np.inf, -np.inf], np.nan).notna()
        & design.notna().all(axis=1)
        & regression_weights.notna()
        & regression_weights.gt(0)
    )
    _warn_dropped_tickers(tickers, valid)
    if valid.sum() <= design.shape[1]:
        raise ValueError(
            "Not enough complete securities to estimate the factor returns."
        )

    regression_design = design.loc[valid].astype(float)
    regression_returns = stock_returns.loc[valid].astype(float)
    weights = regression_weights.loc[valid].astype(float)
    root_weights = np.sqrt(weights)
    weighted_design = regression_design.mul(root_weights, axis=0)
    weighted_returns = regression_returns.mul(root_weights)
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design.to_numpy(),
        weighted_returns.to_numpy(),
        rcond=None,
    )
    if rank < regression_design.shape[1]:
        print(
            f"WARNING: Regression matrix rank is {rank}, below its "
            f"{regression_design.shape[1]} columns.",
            file=sys.stderr,
        )

    factor_returns = pd.Series(
        coefficients,
        index=regression_design.columns,
        name="factor_return",
    )
    residuals = regression_returns - regression_design.dot(factor_returns)
    weighted_mean = np.average(regression_returns, weights=weights)
    residual_sum_squares = weights.dot(residuals.pow(2))
    total_sum_squares = weights.dot((regression_returns - weighted_mean).pow(2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    return FactorReturnResult(
        exposure_date=exposure_date,
        return_date=return_date,
        factor_returns=factor_returns,
        observations=int(valid.sum()),
        r_squared=float(r_squared),
        rank=int(rank),
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Estimate factor returns for a YYYYMMDD return date."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "date",
        help="Return date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--port-data-dir",
        type=Path,
        default=Path("port_data"),
        help="Directory containing estimation-universe workbooks.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("bloomberg_history.db"),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV destination for the estimated factor returns.",
    )
    options = parser.parse_args(arguments)
    return_date = datetime.strptime(options.date, "%Y%m%d").date()

    result = estimate_factor_returns(
        return_date,
        options.port_data_dir,
        options.database,
    )
    output = result.factor_returns.rename_axis("factor").reset_index()
    output.insert(0, "return_date", result.return_date.isoformat())
    output.insert(0, "exposure_date", result.exposure_date.isoformat())
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(options.output, index=False)
        print(f"Saved factor returns to {options.output}.")

    print(
        f"Exposure date: {result.exposure_date}; return date: "
        f"{result.return_date}; observations: {result.observations:,}; "
        f"weighted R-squared: {result.r_squared:.6f}"
    )
    display = result.factor_returns.to_frame("return")
    display["return_percent"] = display["return"] * 100.0
    print(display.to_string(float_format=lambda value: f"{value:.8f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
