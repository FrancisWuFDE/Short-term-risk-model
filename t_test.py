"""Test portfolio factor exposures over time against a zero mean."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

DATE_ROW_PATTERN = re.compile(
    r"^Exposure date: (?P<exposure>\d{4}-\d{2}-\d{2}); "
    r"return date: (?P<return>\d{4}-\d{2}-\d{2})\.$"
)
EXPOSURE_ROW_PATTERN = re.compile(
    r"^\s*(?P<factor>[A-Za-z][A-Za-z0-9_]*)\s+"
    r"(?P<portfolio>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"(?P<benchmark>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def load_exposure_history(log_path: str | Path) -> pd.DataFrame:
    """Parse each portfolio-exposure table in a factor exposure log."""
    rows = []
    current_row: dict[str, float | str] | None = None
    pending_dates: dict[str, str] = {}
    with Path(log_path).open(encoding="utf-8") as log_file:
        for line in log_file:
            date_match = DATE_ROW_PATTERN.fullmatch(line.strip())
            if date_match:
                pending_dates = {
                    "exposure_date": date_match.group("exposure"),
                    "return_date": date_match.group("return"),
                }
                continue
            if "portfolio_exposure" in line and "benchmark_exposure" in line:
                if current_row:
                    rows.append(current_row)
                current_row = pending_dates.copy()
                pending_dates = {}
                continue
            if current_row is None:
                continue

            match = EXPOSURE_ROW_PATTERN.fullmatch(line.rstrip())
            if match:
                current_row[match.group("factor")] = float(
                    match.group("portfolio")
                )

    if current_row:
        rows.append(current_row)
    if not rows:
        raise ValueError(f"No factor exposure tables found in {log_path}.")

    history = pd.DataFrame(rows)
    for column in ("exposure_date", "return_date"):
        if column in history:
            history[column] = pd.to_datetime(history[column])
    history.index = pd.RangeIndex(1, len(history) + 1, name="observation")
    return history


def newey_west_mean_test(
    values: pd.Series,
    confidence_level: float = 0.95,
    max_lags: int | None = None,
) -> pd.Series:
    """Test whether a time-series mean is zero using a Newey-West error."""
    observations = values.dropna().astype(float)
    count = len(observations)
    if count < 2:
        raise ValueError("At least two observations are required for a t-test.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one.")

    if max_lags is None:
        max_lags = int(np.floor(4.0 * (count / 100.0) ** (2.0 / 9.0)))
    if not 0 <= max_lags < count:
        raise ValueError("max_lags must be between zero and observations - 1.")

    residuals = observations.to_numpy() - observations.mean()
    long_run_variance = np.dot(residuals, residuals) / count
    for lag in range(1, max_lags + 1):
        covariance = np.dot(residuals[lag:], residuals[:-lag]) / count
        bartlett_weight = 1.0 - lag / (max_lags + 1.0)
        long_run_variance += 2.0 * bartlett_weight * covariance

    standard_error = np.sqrt(max(long_run_variance, 0.0) / count)
    mean_exposure = observations.mean()
    t_statistic = mean_exposure / standard_error
    degrees_of_freedom = count - 1
    alpha = 1.0 - confidence_level
    p_value = 2.0 * stats.t.sf(abs(t_statistic), degrees_of_freedom)
    critical_value = stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom)
    margin = critical_value * standard_error
    return pd.Series(
        {
            "mean_exposure": mean_exposure,
            "standard_error": standard_error,
            "t_stat": t_statistic,
            "p_value": p_value,
            "confidence_lower": mean_exposure - margin,
            "confidence_upper": mean_exposure + margin,
            "significant_95pct": p_value < alpha,
            "observations": count,
            "max_lags": max_lags,
        }
    )


def test_factor_exposure_history(
    exposure_history: pd.DataFrame,
    confidence_level: float = 0.95,
    max_lags: int | None = None,
) -> pd.DataFrame:
    """Run a zero-mean time-series test for each factor exposure."""
    metadata_columns = {"exposure_date", "return_date"}
    factor_columns = exposure_history.columns.difference(metadata_columns)
    results = {
        factor: newey_west_mean_test(
            exposure_history[factor],
            confidence_level,
            max_lags,
        )
        for factor in factor_columns
    }
    return pd.DataFrame(results).T.rename_axis("factor")


def main(arguments: Sequence[str] | None = None) -> int:
    """Test the factor exposure history stored in a log file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("port_outputs/factor_exposures.log"),
        help="Factor exposure log path.",
    )
    parser.add_argument(
        "--max-lags",
        type=int,
        help="Newey-West lag count; defaults to an automatic sample-size rule.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV destination for the test results.",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        help="Optional CSV destination for the daily exposure history.",
    )
    options = parser.parse_args(arguments)
    history = load_exposure_history(options.log)
    if len(history) < 20:
        print(
            f"WARNING: Only {len(history)} exposure observations are available; "
            "the time-series significance results will be unreliable.",
            file=sys.stderr,
        )
    results = test_factor_exposure_history(
        history,
        max_lags=options.max_lags,
    )
    print(results.to_string(float_format=lambda value: f"{value:.8f}"))

    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        results.reset_index().to_csv(options.output, index=False)
        print(f"Saved significance results to {options.output}.")
    if options.history_output:
        options.history_output.parent.mkdir(parents=True, exist_ok=True)
        history.reset_index().to_csv(options.history_output, index=False)
        print(f"Saved exposure history to {options.history_output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
