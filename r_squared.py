"""Relate dated portfolio factor exposures to subsequent portfolio returns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from factor_returns import load_stock_returns
from short_term_factors import (
    calculate_portfolio_weights,
    load_latest_prices,
    load_portfolio_positions,
)
from t_test import load_exposure_history


def calculate_portfolio_return_history(
    exposure_history: pd.DataFrame,
    port_data_dir: str | Path = "port_data",
    database_path: str | Path = "bloomberg_history.db",
    portfolio_prefix: str = "US_live_port",
) -> pd.DataFrame:
    """Calculate gross-normalized returns for each dated exposure observation."""
    required_columns = {"exposure_date", "return_date"}
    if not required_columns.issubset(exposure_history.columns):
        raise ValueError(
            "The exposure log has no dates. Regenerate it with the updated "
            "short_term_factors.py before calculating R-squared."
        )

    rows = []
    data_dir = Path(port_data_dir)
    dated_history = exposure_history.dropna(subset=list(required_columns))
    for observation, row in dated_history.iterrows():
        exposure_date = row["exposure_date"].date()
        return_date = row["return_date"].date()
        portfolio_path = data_dir / (
            f"{portfolio_prefix}_{exposure_date:%Y%m%d}.csv"
        )
        _, shares = load_portfolio_positions(portfolio_path)
        prices = load_latest_prices(
            database_path,
            shares.index,
            exposure_date,
        )
        weights = calculate_portfolio_weights(shares, prices)
        stock_returns = load_stock_returns(
            database_path,
            weights.index,
            exposure_date,
            return_date,
        )
        valid = weights.notna() & stock_returns.notna()
        portfolio_return = weights[valid].dot(stock_returns[valid])
        gross_weight_coverage = weights[valid].abs().sum()
        print(
            f"{exposure_date} to {return_date}: return "
            f"{portfolio_return:.8f}; gross coverage "
            f"{gross_weight_coverage:.4%}."
        )
        rows.append(
            {
                "observation": observation,
                "exposure_date": exposure_date,
                "return_date": return_date,
                "portfolio_return": portfolio_return,
                "gross_weight_coverage": gross_weight_coverage,
            }
        )

    if len(rows) < 2:
        raise ValueError("At least two dated exposure observations are required.")
    return pd.DataFrame(rows).set_index("observation")


def calculate_factor_r_squared(
    exposure_history: pd.DataFrame,
    portfolio_returns: pd.Series,
) -> pd.DataFrame:
    """Run one univariate portfolio-return regression for each factor."""
    metadata_columns = {"exposure_date", "return_date"}
    factor_columns = exposure_history.columns.difference(metadata_columns)
    rows = []
    for factor in factor_columns:
        regression_data = pd.concat(
            [
                exposure_history[factor].rename("exposure"),
                portfolio_returns.rename("portfolio_return"),
            ],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(regression_data) < 2:
            print(
                f"WARNING: Skipping {factor}; fewer than two observations.",
                file=sys.stderr,
            )
            continue
        if regression_data["exposure"].nunique() < 2:
            print(
                f"WARNING: Skipping {factor}; exposure does not vary.",
                file=sys.stderr,
            )
            continue

        regression = stats.linregress(
            regression_data["exposure"],
            regression_data["portfolio_return"],
        )
        observations = len(regression_data)
        r_squared = regression.rvalue**2
        adjusted_r_squared = (
            1.0 - (1.0 - r_squared) * (observations - 1) / (observations - 2)
            if observations > 2
            else np.nan
        )
        rows.append(
            {
                "factor": factor,
                "r_squared": r_squared,
                "adjusted_r_squared": adjusted_r_squared,
                "correlation": regression.rvalue,
                "slope": regression.slope,
                "intercept": regression.intercept,
                "p_value": regression.pvalue,
                "observations": observations,
            }
        )

    if not rows:
        raise ValueError("No factor had enough data for an R-squared regression.")
    return pd.DataFrame(rows).set_index("factor").sort_values(
        "r_squared",
        ascending=False,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Calculate each factor exposure's R-squared to portfolio returns."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("port_outputs/factor_exposures.log"),
        help="Dated factor exposure log path.",
    )
    parser.add_argument(
        "--prefix",
        default="US_live_port",
        help=(
            "Portfolio filename prefix before _YYYYMMDD.csv "
            "(default: US_live_port)."
        ),
    )
    parser.add_argument(
        "--port-data-dir",
        type=Path,
        default=Path("port_data"),
        help="Directory containing dated portfolio files.",
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
        help="Optional CSV destination for the R-squared results.",
    )
    parser.add_argument(
        "--returns-output",
        type=Path,
        help="Optional CSV destination for calculated portfolio returns.",
    )
    options = parser.parse_args(arguments)

    exposures = load_exposure_history(options.log)
    returns = calculate_portfolio_return_history(
        exposures,
        options.port_data_dir,
        options.database,
        options.prefix,
    )
    if len(returns) < 20:
        print(
            f"WARNING: Only {len(returns)} dated observations are available; "
            "the R-squared estimates will be unreliable.",
            file=sys.stderr,
        )
    results = calculate_factor_r_squared(
        exposures,
        returns["portfolio_return"],
    )
    print(results.to_string(float_format=lambda value: f"{value:.8f}"))

    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        results.reset_index().to_csv(options.output, index=False)
        print(f"Saved R-squared results to {options.output}.")
    if options.returns_output:
        options.returns_output.parent.mkdir(parents=True, exist_ok=True)
        returns.reset_index().to_csv(options.returns_output, index=False)
        print(f"Saved portfolio returns to {options.returns_output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
