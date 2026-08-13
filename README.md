# Bloomberg PORT Risk-Matrix Workflow

This guide explains how to load a new equity portfolio into Bloomberg PORT,
create the portfolio-versus-benchmark risk output, export the estimation
universe, and run the local short-term risk-factor process.

The Bloomberg portion uses standard Terminal functions available through
`PRTU <GO>` and `PORT <GO>`. It does not require PORT Enterprise, `PREP`, or
`JMGR`.

> Bloomberg periodically changes labels and screen layouts. If a button name
> differs, use the equivalent action in the current PORT workspace and update
> this guide after confirming it.

## Current configuration

The current US workflow uses these settings:

| Setting | Value |
| --- | --- |
| Portfolio | `US_LIVE_PORT` |
| Benchmark | `B3000` |
| Currency | `USD` |
| Classification | `GICS Sector` |
| Risk model | `Integrated Multi-Asset` |
| Model version | `MAC3` |
| Horizon | `Quarterly` |
| Display units | `Returns (%)` |
| Scaling | `1 Year` |

Change these values only when the new analysis requires a different portfolio,
benchmark, model, or reporting convention. Record any change with the output.

## Required files and naming

Put all dated inputs and PORT exports in `port_data`.

```text
port_data/
├── US_live_port_YYYYMMDD.csv
└── estimation_universe_YYYYMMDD.xlsx
```

`YYYYMMDD` is the rebalance date used by the local scripts. For example:

```text
US_live_port_20260129.csv
estimation_universe_20260129.xlsx
```

The PORT workbook can display the following holding date internally. That is
expected in this workflow: positions established at the end of the rebalance
date are the holdings for the following trading day. Do not rename the files to
the internal PORT date.

### Portfolio CSV format

The CSV must contain these columns:

```csv
Date,Port,ticker,shares
02/02/2026,US_live_port,FISV,-2157
02/02/2026,US_live_port,UPBD,4300
```

- `Date`: portfolio holding date in `MM/DD/YYYY` format.
- `Port`: portfolio name.
- `ticker`: base US equity ticker without `US Equity`.
- `shares`: signed share quantity; negative values are short positions.

Before uploading, confirm that there are no blank tickers, duplicate ticker
rows, nonnumeric share quantities, or accidental zero positions. If duplicates
are intentional, aggregate them to one row per ticker first.

## Recurring checklist

For an established portfolio, the recurring process is:

1. Prepare and validate `US_live_port_YYYYMMDD.csv`.
2. Update the dated portfolio in `PRTU <GO>` and save it.
3. Open the portfolio in `PORT <GO>` with the correct benchmark and as-of date.
4. Confirm the model and reporting settings in the table above.
5. Generate the security-level portfolio-and-benchmark matrix.
6. Export it to Excel as `port_data/estimation_universe_YYYYMMDD.xlsx`.
7. Validate that the workbook contains `Ticker / All` and `Ticker / Bmrk`.
8. Run `bbg_cache.py` to update local Bloomberg data.
9. Run `short_term_factors.py` to calculate normalized exposures.
10. Review coverage warnings and confirm benchmark exposures are approximately
    zero.

The detailed procedure follows.

## 1. Prepare the portfolio input

1. Copy the new holdings into the portfolio CSV template.
2. Use the rebalance date in the filename.
3. Preserve signed shares: longs are positive and shorts are negative.
4. Save the file as CSV, not as an Excel workbook renamed to `.csv`.
5. Close the CSV and any prior PORT output workbooks before running local
   scripts. An open workbook can cause a Windows `PermissionError`.

Recommended control totals to record before upload:

- Number of distinct tickers.
- Number of long and short positions.
- Sum of long shares and absolute short shares.
- Source file name and rebalance date.

## 2. Create or update the portfolio in PRTU

1. On the Bloomberg Terminal, enter `PRTU <GO>`.
2. Open `US_LIVE_PORT`. If it does not exist, select **Create**, choose the
   appropriate equity asset class, and name it `US_LIVE_PORT`.
3. Select the effective holding date. This should agree with the `Date` column
   in the CSV, even though the local filename uses the rebalance date.
4. Import or paste the ticker and signed-share columns into the holdings grid.
   The exact import control can vary by Terminal layout; Bloomberg also accepts
   copying or dragging adjacent identifier and position columns into PRTU.
5. Review unresolved securities, duplicate mappings, cash entries, and position
   signs.
6. Compare the uploaded security count and control totals with the source CSV.
7. Save the portfolio.

Do not continue until all material positions are resolved. Record any security
that must be omitted or manually mapped.

## 3. Configure the analysis in PORT

1. With the saved portfolio selected, choose **Analyze**, or enter `PORT <GO>`
   and select `US_LIVE_PORT`.
2. Set the benchmark to `B3000`.
3. Set the as-of date to the intended holding date.
4. Open the risk analysis or matrix workspace.
5. Confirm the current configuration:
   - Currency: `USD`
   - Classification: `GICS Sector`
   - Model: `Integrated Multi-Asset`
   - Model version: `MAC3`
   - Horizon: `Quarterly`
   - Display units: `Returns (%)`
   - Scaling: `1 Year`
6. Set the output to security-level detail and include both the combined
   universe and benchmark ticker columns.
7. Refresh or recalculate the analysis after changing any setting.

Before exporting, verify that the screen shows the intended portfolio,
benchmark, as-of date, and model. A technically valid export with the wrong
date or benchmark is not usable.

## 4. Export the matrix and estimation universe

1. In PORT, use the available **Export**, **Actions**, or **Generate Report**
   control and select Excel output.
2. Export the security-level matrix, not only the portfolio summary.
3. Confirm that the workbook header records the expected analysis settings.
4. Confirm that the ticker section contains:
   - a `Ticker` heading;
   - an `All` subheading containing every security in the combined universe;
   - a `Bmrk` subheading identifying benchmark members.
5. Save the workbook directly as:

   ```text
   port_data/estimation_universe_YYYYMMDD.xlsx
   ```

6. Close Excel before running the scripts.

The local Bloomberg downloader reads every ticker in `Ticker / All`. The factor
model uses `Ticker / Bmrk` as the estimation universe. Therefore, both columns
must be present even if the workbook contains additional PORT results.

## 5. Set up Python on a new machine

From PowerShell in the repository:

```powershell
& "C:\Path\To\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The Bloomberg Terminal must be open and logged in on the same machine. The
Bloomberg Desktop API Python package must also be installed for that Python
environment.

Do not commit `.venv` or `bloomberg_history.db` to Git. The database contains
downloaded data and can exceed GitHub's recommended file-size limit.

## 6. Download and cache Bloomberg data

With the virtual environment active and Bloomberg running:

```powershell
python bbg_cache.py ".\port_data\estimation_universe_YYYYMMDD.xlsx"
```

The script stores data in `bloomberg_history.db` by default. It retrieves:

- Closing prices for the six months ending on the rebalance date.
- Closing prices for the matching one-month period in each of the previous
  five years.
- `SI_PERCENT_EQUITY_FLOAT` from the most recent available date within the
  preceding 21 calendar days.
- Market capitalization for the rebalance date.
- Bloomberg ticker and BICS industry metadata.

The cache checks existing database coverage before making Bloomberg requests.
Completed batches are stored immediately, so a failed run can be restarted.
Securities already covered are skipped. Bloomberg resolution or field errors
are printed and the affected security is omitted or retried on a later run.

Expected checks:

- Loaded security count is plausible for `Ticker / All`.
- Resolved count is close to loaded count.
- Cached securities and windows are reported as skipped on repeat runs.
- Any unresolved or missing-data warnings are reviewed.

To use a different database path:

```powershell
python bbg_cache.py ".\port_data\estimation_universe_YYYYMMDD.xlsx" `
    --database ".\bloomberg_history.db"
```

## 7. Calculate short-term factor exposures

Run:

```powershell
python short_term_factors.py YYYYMMDD
```

The script automatically loads:

- `port_data/US_live_port_YYYYMMDD.csv` for portfolio positions;
- `port_data/estimation_universe_YYYYMMDD.xlsx` for benchmark tickers; and
- `bloomberg_history.db` for Bloomberg data.

It currently calculates:

- One-day reversal.
- Short-term reversal.
- Seasonality.
- Industry momentum.
- Short interest.
- Downside risk.

Raw descriptors are winsorized, normalized across the benchmark estimation
universe, and shifted so that the cap-weighted benchmark exposure is zero. The
portfolio exposure is the sum of each normalized stock exposure multiplied by
its portfolio weight.

## 8. Validate and retain the results

Review the console output before using the results:

1. **Portfolio price coverage** should be close to the number of positions.
   Investigate missing portfolio securities.
2. **Benchmark exposure** for every normalized factor should be zero within
   floating-point tolerance, such as `0.00000000` or `-0.00000000`.
3. **Portfolio exposure** should be finite. A missing or infinite value signals
   insufficient input data or a degenerate estimation distribution.
4. Review Bloomberg warnings for unresolved tickers and stale or missing
   point-in-time data.
5. Retain the dated CSV, PORT Excel export, console output, and code version
   used for the run.

A suggested output convention is:

```text
results/
└── factor_exposures_YYYYMMDD.txt
```

Until file output is automated, PowerShell can capture the console output:

```powershell
New-Item -ItemType Directory -Force ".\results" | Out-Null
python short_term_factors.py YYYYMMDD |
    Tee-Object ".\results\factor_exposures_YYYYMMDD.txt"
```

## Automation available without PREP or JMGR

The local portion is already mostly automated. The remaining manual boundary
is loading the holdings into Bloomberg PORT and exporting its workbook.

### Available now

- Standard dated file discovery.
- Bloomberg Desktop API data retrieval.
- SQLite caching and restart after completed batches.
- Factor calculation, winsorization, normalization, and benchmark-neutrality
  checks.
- Console-result capture with `Tee-Object`.

### Recommended next automation

Add one local command, for example `run_risk_matrix.py YYYYMMDD`, that:

1. Validates the CSV and exported workbook structure.
2. Checks that dates and expected PORT settings agree.
3. Runs `bbg_cache.py`.
4. Runs `short_term_factors.py`.
5. Writes a dated CSV or Excel result and a validation log.

This would automate everything after the PORT Excel export while staying within
the available Desktop Terminal permissions.

### Terminal-side automation limitation

The Bloomberg Desktop API used by `bbg_cache.py` retrieves reference and
historical data; it does not automate interactive PRTU/PORT portfolio uploads or
screen exports. Without PORT Enterprise reporting access, keep those steps
manual unless Bloomberg enables another licensed upload interface for the
account. Do not automate the Terminal UI with keystrokes or screen scraping for
a production process; it is fragile and difficult to audit.

## Troubleshooting

### `PermissionError` for the estimation-universe workbook

Close the workbook in Excel and rerun the command.

### Bloomberg `blpapi` package is missing

Install Bloomberg's Python package into the active virtual environment, then
run with the Bloomberg Terminal open and logged in.

### Security cannot be resolved

Check the ticker and exchange mapping in PORT. The downloader prints the
identifier and skips unresolved securities rather than stopping the full run.

### Repeat run still downloads some data

The database only skips a security/window when complete coverage is recorded.
Incomplete histories and prior Bloomberg errors are intentionally requested
again.

### Benchmark exposure is not approximately zero

Check benchmark market-cap coverage, missing descriptors, the exported `Bmrk`
column, and whether the correct as-of date was used.

## Items to confirm during walkthrough

The following should be verified on the actual Bloomberg account and then
updated in this guide:

- Exact PRTU import button/menu name used for the equity CSV.
- Exact PORT workspace/tab used to build the risk matrix.
- Exact Excel export menu name.
- Whether the saved PORT layout can be reused for each new date.
- Final required output fields beyond the estimation-universe ticker columns.
- Desired permanent format and location for the calculated risk matrix.

## References

- [Bloomberg Portfolio & Risk Analytics](https://professional.bloomberg.com/products/bloomberg-terminal/portfolio-analytics/)
- [Bloomberg example using PRTU and PORT](https://www.bloomberg.com/professional/insights/trading/evaluate-portfolio-trades-efficiently-with-port-and-fiw/)
