"""Fit a zero-mean GARCH(1,1) model to portfolio returns."""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from r_squared import calculate_portfolio_return_history
from t_test import load_exposure_history

MINIMUM_OBSERVATIONS = 3
VERY_SMALL_SAMPLE = 30
RECOMMENDED_OBSERVATIONS = 250
TRADING_DAYS_PER_YEAR = 252


def load_portfolio_returns(returns_path: str | Path) -> pd.Series:
    """Load a CSV containing return_date and portfolio_return columns."""
    returns = pd.read_csv(returns_path)
    required_columns = {"return_date", "portfolio_return"}
    missing_columns = required_columns.difference(returns.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Returns file is missing columns: {missing}.")

    returns["return_date"] = pd.to_datetime(returns["return_date"])
    series = returns.set_index("return_date")["portfolio_return"]
    return series.astype(float).sort_index().rename("portfolio_return")


def download_spy(start_date: str, end_date: str) -> pd.Series:
    """Download SPY returns from Yahoo Finance."""
    spy_data = yf.download(
        "SPY",
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )
    if spy_data.empty:
        raise ValueError("Yahoo Finance returned no SPY prices.")
    adjusted_close = spy_data["Adj Close"]
    if isinstance(adjusted_close, pd.DataFrame):
        adjusted_close = adjusted_close.iloc[:, 0]
    spy_returns = adjusted_close.pct_change(fill_method=None).dropna()
    return spy_returns.rename("spy_return")


def download_four_year_spy_returns(as_of_date: date) -> pd.Series:
    """Download four years of SPY returns through an inclusive as-of date."""
    end_date = as_of_date + timedelta(days=1)
    start_date = (
        pd.Timestamp(as_of_date) - pd.DateOffset(years=4)
    ).date()
    returns = download_spy(
        start_date.isoformat(),
        end_date.isoformat(),
    )
    print(
        f"Loaded {len(returns):,} SPY returns from "
        f"{returns.index.min().date()} through {returns.index.max().date()}."
    )
    return returns


def prepare_portfolio_returns(portfolio_returns: pd.Series) -> pd.Series:
    """Clean and validate a chronological portfolio-return series."""
    returns = portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()
    returns = returns.groupby(level=0).last().sort_index().astype(float)
    if len(returns) < MINIMUM_OBSERVATIONS:
        raise ValueError(
            f"GARCH requires at least {MINIMUM_OBSERVATIONS} returns; "
            f"only {len(returns)} are available."
        )
    if returns.nunique() < 2:
        raise ValueError("Portfolio returns must vary over time.")
    return returns


def fit_garch_11(
    portfolio_returns: pd.Series,
    horizon: int = 1,
    distribution: str = "t",
) -> tuple[pd.DataFrame, pd.Series, Any]:
    """Fit zero-mean GARCH(1,1) and forecast conditional volatility."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    returns = prepare_portfolio_returns(portfolio_returns)

    try:
        from arch import arch_model
    except ImportError as error:
        raise RuntimeError(
            "The arch package is required. Run: pip install arch"
        ) from error

    percentage_returns = returns * 100.0
    model = arch_model(
        percentage_returns,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist=distribution,
        rescale=False,
    )
    result = model.fit(disp="off")
    variance_forecast = result.forecast(
        horizon=horizon,
        reindex=False,
    ).variance.iloc[-1]
    daily_volatility = np.sqrt(variance_forecast) / 100.0
    forecasts = pd.DataFrame(
        {
            "horizon": range(1, horizon + 1),
            "daily_variance": variance_forecast.to_numpy() / 10_000.0,
            "daily_volatility": daily_volatility.to_numpy(),
            "annualized_volatility": (
                daily_volatility.to_numpy()
                * np.sqrt(TRADING_DAYS_PER_YEAR)
            ),
        }
    ).set_index("horizon")

    omega = float(result.params["omega"])
    alpha = float(result.params["alpha[1]"])
    beta = float(result.params["beta[1]"])
    persistence = alpha + beta
    long_run_volatility = (
        np.sqrt(omega / (1.0 - persistence)) / 100.0
        if persistence < 1.0 - 1e-6
        else np.nan
    )
    diagnostics = pd.Series(
        {
            "observations": len(returns),
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "persistence": persistence,
            "long_run_daily_volatility": long_run_volatility,
            "long_run_annualized_volatility": (
                long_run_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
            ),
            "log_likelihood": result.loglikelihood,
            "aic": result.aic,
            "bic": result.bic,
            "convergence_flag": result.convergence_flag,
        },
        name="value",
    )
    if "nu" in result.params:
        diagnostics.loc["student_t_degrees_of_freedom"] = result.params["nu"]
    return forecasts, diagnostics, result


def calculate_rolling_garch_forecasts(
    returns: pd.Series,
    training_years: int = 2,
    forecast_years: int = 2,
    distribution: str = "t",
) -> pd.DataFrame:
    """Generate one-day forecasts using a trailing, no-lookahead window."""
    if training_years <= 0 or forecast_years <= 0:
        raise ValueError("training_years and forecast_years must be positive.")
    returns = prepare_portfolio_returns(returns)
    forecast_start = returns.index.max() - pd.DateOffset(years=forecast_years)
    forecast_dates = returns.index[returns.index >= forecast_start]
    rows = []

    for position, forecast_date in enumerate(forecast_dates, start=1):
        training_start = forecast_date - pd.DateOffset(years=training_years)
        training_returns = returns[
            (returns.index >= training_start)
            & (returns.index < forecast_date)
        ]
        if len(training_returns) < RECOMMENDED_OBSERVATIONS:
            print(
                f"WARNING: Skipping {forecast_date.date()}; only "
                f"{len(training_returns)} training returns are available.",
                file=sys.stderr,
            )
            continue

        forecast, diagnostics, _ = fit_garch_11(
            training_returns,
            horizon=1,
            distribution=distribution,
        )
        forecast_row = forecast.iloc[0]
        rows.append(
            {
                "forecast_date": forecast_date,
                "training_start": training_returns.index.min(),
                "training_end": training_returns.index.max(),
                "training_observations": len(training_returns),
                "realized_return": returns.loc[forecast_date],
                "forecast_daily_variance": forecast_row["daily_variance"],
                "forecast_daily_volatility": forecast_row["daily_volatility"],
                "forecast_annualized_volatility": forecast_row[
                    "annualized_volatility"
                ],
                "omega": diagnostics["omega"],
                "alpha": diagnostics["alpha"],
                "beta": diagnostics["beta"],
                "persistence": diagnostics["persistence"],
                "convergence_flag": diagnostics["convergence_flag"],
            }
        )
        if position == 1 or position % 25 == 0 or position == len(forecast_dates):
            print(
                f"Rolling GARCH forecast {position}/{len(forecast_dates)}: "
                f"{forecast_date.date()}."
            )

    if not rows:
        raise ValueError("No rolling GARCH forecasts could be calculated.")
    return pd.DataFrame(rows).set_index("forecast_date")


def main(arguments: Sequence[str] | None = None) -> int:
    """Fit GARCH(1,1) and print forward volatility forecasts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=("spy", "portfolio"),
        default="spy",
        help="Return series to model (default: spy).",
    )
    parser.add_argument(
        "--mode",
        choices=("rolling", "latest"),
        default="rolling",
        help="Two-year rolling history or one latest forecast (default: rolling).",
    )
    parser.add_argument(
        "--as-of-date",
        help="Inclusive SPY end date in YYYYMMDD format (default: today).",
    )
    parser.add_argument(
        "--returns-file",
        type=Path,
        help="CSV containing return_date and portfolio_return columns.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("port_outputs/factor_exposures.log"),
        help="Dated exposure log used when --returns-file is omitted.",
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
        "--horizon",
        type=int,
        default=1,
        help="Number of future daily volatility forecasts (default: 1).",
    )
    parser.add_argument(
        "--training-years",
        type=int,
        default=2,
        help="Trailing years used for each rolling fit (default: 2).",
    )
    parser.add_argument(
        "--forecast-years",
        type=int,
        default=2,
        help="Years of rolling forecasts to produce (default: 2).",
    )
    parser.add_argument(
        "--distribution",
        choices=("t", "normal"),
        default="t",
        help="Standardized innovation distribution (default: t).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("port_outputs/garch_volatility_forecasts.csv"),
        help="CSV destination for volatility forecasts.",
    )
    parser.add_argument(
        "--returns-output",
        type=Path,
        help="Optional CSV destination for the return history used.",
    )
    options = parser.parse_args(arguments)

    try:
        if options.returns_file:
            portfolio_returns = load_portfolio_returns(options.returns_file)
        elif options.source == "spy":
            as_of_date = (
                datetime.strptime(options.as_of_date, "%Y%m%d").date()
                if options.as_of_date
                else date.today()
            )
            portfolio_returns = download_four_year_spy_returns(as_of_date)
        else:
            exposures = load_exposure_history(options.log)
            return_history = calculate_portfolio_return_history(
                exposures,
                options.port_data_dir,
                options.database,
                options.prefix,
            )
            portfolio_returns = return_history.set_index("return_date")[
                "portfolio_return"
            ]

        if options.mode == "rolling":
            forecasts = calculate_rolling_garch_forecasts(
                portfolio_returns,
                options.training_years,
                options.forecast_years,
                options.distribution,
            )
            diagnostics = None
        else:
            observation_count = len(portfolio_returns.dropna())
            if observation_count < VERY_SMALL_SAMPLE:
                print(
                    f"WARNING: Only {observation_count} returns are available. "
                    "This GARCH forecast is extremely unreliable and intended "
                    "only as a provisional calculation.",
                    file=sys.stderr,
                )
            elif observation_count < RECOMMENDED_OBSERVATIONS:
                print(
                    f"WARNING: Fewer than {RECOMMENDED_OBSERVATIONS} returns "
                    "are available; GARCH estimates may be unstable.",
                    file=sys.stderr,
                )
            forecasts, diagnostics, _ = fit_garch_11(
                portfolio_returns,
                options.horizon,
                options.distribution,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if diagnostics is not None:
        print("Model diagnostics:")
        print(diagnostics.to_string(float_format=lambda value: f"{value:.8f}"))
        print("\nVolatility forecasts:")
        print(forecasts.to_string(float_format=lambda value: f"{value:.8f}"))
    else:
        print(
            f"Calculated {len(forecasts):,} rolling forecasts from "
            f"{forecasts.index.min().date()} through "
            f"{forecasts.index.max().date()}."
        )
        print(forecasts.tail().to_string(float_format=lambda value: f"{value:.8f}"))

    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        forecasts.reset_index().to_csv(options.output, index=False)
        print(f"Saved volatility forecasts to {options.output}.")
    if options.returns_output:
        options.returns_output.parent.mkdir(parents=True, exist_ok=True)
        output_returns = portfolio_returns.rename(
            "portfolio_return"
        ).rename_axis("return_date").reset_index()
        output_returns.to_csv(options.returns_output, index=False)
        print(f"Saved portfolio returns to {options.returns_output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
