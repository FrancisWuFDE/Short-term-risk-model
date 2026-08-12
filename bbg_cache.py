"""Load Bloomberg price and short-interest data into a SQLite database.

For ``estimation_universe_YYYYMMDD.xlsx`` inputs, the filename supplies the
rebalance date and the ``Ticker / All`` column supplies the securities.
Bloomberg Terminal and the Desktop API must run on the same machine.
"""

from __future__ import annotations

import argparse
import calendar
import importlib
import math
import re
import sqlite3
import sys
import zipfile
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypeVar
from xml.etree import ElementTree


REFDATA_SERVICE = "//blp/refdata"
PRICE_FIELD = "PX_LAST"
TICKER_FIELD = "TICKER_AND_EXCH_CODE"
BICS_INDUSTRY_FIELD = "BICS_LEVEL_3_INDUSTRY_NAME"
SHORT_INTEREST_FIELD = "SI_PERCENT_EQUITY_FLOAT"
MARKET_CAP_FIELD = "CUR_MKT_CAP"
MARKET_CAP_LOOKBACK_DAYS = 31

FIGI_PATTERN = re.compile(r"^BBG[A-Z0-9]{9}$", re.IGNORECASE)
FIGI_HEADERS = {
    "figi",
    "bbgid",
    "bloomberg global id",
    "bloomberg global identifier",
}
TICKER_HEADERS = {"ticker", "tickers"}
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9./*^_-]*$", re.IGNORECASE)
AS_OF_HEADERS = {"as of date", "as-of date", "asof date"}
ESTIMATION_UNIVERSE_FILENAME = re.compile(
    r"estimation_universe_(\d{8})",
    re.IGNORECASE,
)

MAIN_XML_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_XML_NAMESPACE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_REL_NAMESPACE = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
T = TypeVar("T")


@dataclass(frozen=True)
class WorkbookInput:
    """Inputs parsed from the source workbook."""

    as_of_date: date
    identifiers: tuple[str, ...]


@dataclass(frozen=True)
class SecurityInfo:
    """Bloomberg metadata for one valid input security."""

    input_identifier: str
    bloomberg_security: str
    ticker: str
    industry: str


@dataclass(frozen=True)
class PriceRow:
    """One historical closing-price observation."""

    observation_date: date
    ticker: str
    industry: str
    price: float


@dataclass(frozen=True)
class ShortInterestRow:
    """Latest available short-interest observation for one security."""

    observation_date: date
    ticker: str
    industry: str
    short_interest_percent_float: float


@dataclass(frozen=True)
class MarketCapRow:
    """Latest available market-cap observation for one security."""

    observation_date: date
    ticker: str
    industry: str
    market_cap: float


def _load_blpapi() -> Any:
    """Import Bloomberg's Python package or raise an actionable error."""
    try:
        return importlib.import_module("blpapi")
    except ImportError as error:
        raise RuntimeError(
            "Bloomberg's blpapi package is required. Install it from the "
            "Bloomberg package index and run with Bloomberg Terminal open."
        ) from error


def _column_number(cell_reference: str) -> int:
    """Convert an Excel cell reference to a one-based column number."""
    letters = "".join(character for character in cell_reference if character.isalpha())
    number = 0
    for character in letters.upper():
        number = number * 26 + ord(character) - ord("A") + 1
    return number


def _excel_serial_to_date(value: str) -> date | None:
    """Convert a plausible Excel date serial to a date."""
    try:
        serial = float(value)
    except (TypeError, ValueError):
        return None
    if not 1 <= serial <= 200_000:
        return None
    return date(1899, 12, 30) + timedelta(days=math.floor(serial))


def _parse_date(value: str) -> date | None:
    """Parse common workbook date representations."""
    cleaned = value.strip()
    for date_format in (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y/%m/%d",
        "%d-%b-%Y",
    ):
        try:
            return datetime.strptime(cleaned, date_format).date()
        except ValueError:
            continue
    return _excel_serial_to_date(cleaned)


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """Return the workbook's shared-string table, if present."""
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []

    namespace = {"m": MAIN_XML_NAMESPACE}
    root = ElementTree.fromstring(archive.read(path))
    return [
        "".join(text.text or "" for text in item.findall(".//m:t", namespace))
        for item in root.findall("m:si", namespace)
    ]


def _worksheet_paths(archive: zipfile.ZipFile) -> list[str]:
    """Resolve workbook sheet relationships to archive paths."""
    main_namespace = {"m": MAIN_XML_NAMESPACE, "r": REL_XML_NAMESPACE}
    relationship_namespace = {"r": PACKAGE_REL_NAMESPACE}
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationships.findall("r:Relationship", relationship_namespace)
    }

    paths: list[str] = []
    relationship_key = f"{{{REL_XML_NAMESPACE}}}id"
    for sheet in workbook.findall("m:sheets/m:sheet", main_namespace):
        target = targets[sheet.attrib[relationship_key]].replace("\\", "/")
        paths.append(target.lstrip("/") if target.startswith("xl/") else f"xl/{target}")
    return paths


def _read_workbook_cells(workbook_path: Path) -> dict[tuple[int, int, int], str]:
    """Read non-empty cells from an XLSX using only the Python standard library."""
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Input workbook not found: {workbook_path}")

    namespace = {"m": MAIN_XML_NAMESPACE}
    cells: dict[tuple[int, int, int], str] = {}
    try:
        with zipfile.ZipFile(workbook_path) as archive:
            strings = _shared_strings(archive)
            for sheet_number, sheet_path in enumerate(
                _worksheet_paths(archive),
                start=1,
            ):
                root = ElementTree.fromstring(archive.read(sheet_path))
                for cell in root.findall(".//m:sheetData/m:row/m:c", namespace):
                    reference = cell.attrib.get("r", "")
                    row_number_text = "".join(
                        character for character in reference if character.isdigit()
                    )
                    if not reference or not row_number_text:
                        continue

                    value_element = cell.find("m:v", namespace)
                    inline_text = cell.find("m:is/m:t", namespace)
                    if inline_text is not None:
                        value = inline_text.text or ""
                    elif value_element is None:
                        value = ""
                    else:
                        value = value_element.text or ""
                        if cell.attrib.get("t") == "s":
                            value = strings[int(value)]

                    cells[
                        (
                            sheet_number,
                            int(row_number_text),
                            _column_number(reference),
                        )
                    ] = value.strip()
    except (KeyError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"Unable to read XLSX workbook: {workbook_path}") from error
    return cells


def read_workbook_input(
    workbook_path: Path,
    ticker_suffix: str = "US Equity",
) -> WorkbookInput:
    """Extract the data date and distinct Bloomberg securities from an XLSX."""
    cells = _read_workbook_cells(workbook_path)
    suffix = " ".join(ticker_suffix.split())
    if not suffix:
        raise ValueError("ticker_suffix cannot be blank")

    filename_match = ESTIMATION_UNIVERSE_FILENAME.fullmatch(workbook_path.stem)
    if filename_match:
        rebalance_date = datetime.strptime(
            filename_match.group(1),
            "%Y%m%d",
        ).date()
        all_header = next(
            (
                (sheet, row, column)
                for (sheet, row, column), value in cells.items()
                if value.casefold() == "all"
                and cells.get((sheet, row - 1, column), "").casefold()
                in TICKER_HEADERS
            ),
            None,
        )
        if all_header is None:
            raise ValueError("No Ticker/All column was found in the workbook.")

        sheet, header_row, column = all_header
        last_row = max(row for cell_sheet, row, _ in cells if cell_sheet == sheet)
        tickers = tuple(
            dict.fromkeys(
                cells.get((sheet, row, column), "").strip().upper()
                for row in range(header_row + 1, last_row + 1)
                if TICKER_PATTERN.fullmatch(
                    cells.get((sheet, row, column), "").strip()
                )
            )
        )
        return WorkbookInput(
            as_of_date=rebalance_date,
            identifiers=tuple(f"{ticker} {suffix}" for ticker in tickers),
        )

    as_of_date: date | None = None

    for (sheet, row, column), value in cells.items():
        if value.casefold() not in AS_OF_HEADERS:
            continue
        candidate_locations = (
            (sheet, row + 1, column),
            (sheet, row, column + 1),
        )
        for location in candidate_locations:
            candidate = _parse_date(cells.get(location, ""))
            if candidate is not None:
                as_of_date = candidate
                break
        if as_of_date is not None:
            break

    if as_of_date is None:
        raise ValueError(
            "No as-of date was found. Add an 'As Of Date' label with the date "
            "in the cell directly below or to its right."
        )

    tickers: list[str] = []
    maximum_rows: dict[int, int] = {}
    for sheet, cell_row, _ in cells:
        maximum_rows[sheet] = max(maximum_rows.get(sheet, 0), cell_row)
    for (sheet, row, column), value in cells.items():
        if value.casefold() not in TICKER_HEADERS:
            continue
        for next_row in range(row + 1, maximum_rows[sheet] + 1):
            candidate = cells.get((sheet, next_row, column), "").strip().upper()
            if candidate and TICKER_PATTERN.fullmatch(candidate):
                tickers.append(candidate)

    distinct_tickers = tuple(dict.fromkeys(tickers))
    if distinct_tickers:
        return WorkbookInput(
            as_of_date=as_of_date,
            identifiers=tuple(f"{ticker} {suffix}" for ticker in distinct_tickers),
        )

    figis: list[str] = []
    for (sheet, row, column), value in cells.items():
        if value.casefold() not in FIGI_HEADERS:
            continue
        next_row = row + 1
        while (sheet, next_row, column) in cells:
            candidate = cells[(sheet, next_row, column)].strip().upper()
            if candidate and FIGI_PATTERN.fullmatch(candidate):
                figis.append(candidate)
            next_row += 1

    if not figis:
        figis = [
            value.upper()
            for value in cells.values()
            if FIGI_PATTERN.fullmatch(value.strip())
        ]

    distinct_figis = tuple(dict.fromkeys(figis))
    if not distinct_figis:
        raise ValueError(
            "No securities were found. Add a column headed 'Ticker', 'FIGI', or "
            "'BBGID'."
        )
    return WorkbookInput(
        as_of_date=as_of_date,
        identifiers=tuple(f"/bbgid/{figi}" for figi in distinct_figis),
    )


def _add_months(value: date, months: int) -> date:
    """Shift a date by whole calendar months, clipping invalid month-end days."""
    month_index = value.year * 12 + value.month - 1 + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _replace_year(value: date, year: int) -> date:
    """Move a date to another year, mapping February 29 to February 28."""
    day = min(value.day, calendar.monthrange(year, value.month)[1])
    return date(year, value.month, day)


def historical_price_windows(as_of_date: date) -> list[tuple[date, date]]:
    """Build the current lookback and five one-month anniversary windows."""
    windows = [(_add_months(as_of_date, -6), as_of_date)]
    for years_ago in range(1, 6):
        anniversary = _replace_year(as_of_date, as_of_date.year - years_ago)
        windows.append((anniversary, _add_months(anniversary, 1)))
    return windows


def _chunks(values: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield fixed-size slices from a sequence."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


class BloombergClient:
    """Small synchronous wrapper around Bloomberg's Desktop API."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8194,
        batch_size: int = 50,
    ) -> None:
        self.blpapi = _load_blpapi()
        options = self.blpapi.SessionOptions()
        options.setServerHost(host)
        options.setServerPort(port)
        self.session = self.blpapi.Session(options)
        self.service: Any | None = None
        self.batch_size = batch_size

    def __enter__(self) -> BloombergClient:
        if not self.session.start():
            raise RuntimeError(
                "Unable to start a Bloomberg Desktop API session. Confirm that "
                "Bloomberg Terminal is open and logged in."
            )
        if not self.session.openService(REFDATA_SERVICE):
            self.session.stop()
            raise RuntimeError(f"Unable to open Bloomberg service {REFDATA_SERVICE}.")
        self.service = self.session.getService(REFDATA_SERVICE)
        return self

    def __exit__(self, *_: object) -> None:
        self.session.stop()

    def _response_messages(self, request: Any) -> Iterator[Any]:
        """Send a request and yield messages until its final response arrives."""
        self.session.sendRequest(request)
        while True:
            event = self.session.nextEvent(500)
            event_type = event.eventType()
            if event_type == self.blpapi.Event.TIMEOUT:
                continue
            if event_type in (
                self.blpapi.Event.PARTIAL_RESPONSE,
                self.blpapi.Event.RESPONSE,
            ):
                for message in event:
                    if message.hasElement("responseError"):
                        raise RuntimeError(
                            f"Bloomberg response error: "
                            f"{message.getElement('responseError')}"
                        )
                    yield message
            elif event_type == self.blpapi.Event.REQUEST_STATUS:
                for message in event:
                    if message.messageType() == self.blpapi.Name("RequestFailure"):
                        raise RuntimeError(f"Bloomberg request failed: {message}")
            if event_type == self.blpapi.Event.RESPONSE:
                return

    @staticmethod
    def _field_value(field_data: Any, field_name: str) -> str | None:
        """Return a non-null Bloomberg field as text."""
        if not field_data.hasElement(field_name):
            return None
        element = field_data.getElement(field_name)
        if element.isNull():
            return None
        return element.getValueAsString().strip()

    def resolve_securities(
        self,
        identifiers: Sequence[str],
    ) -> list[SecurityInfo]:
        """Resolve input identifiers and retrieve ticker and BICS industry."""
        if self.service is None:
            raise RuntimeError("Bloomberg session has not been opened.")

        resolved: list[SecurityInfo] = []
        for identifier_batch in _chunks(identifiers, self.batch_size):
            request = self.service.createRequest("ReferenceDataRequest")
            securities_element = request.getElement("securities")
            for identifier in identifier_batch:
                securities_element.appendValue(identifier)
            fields_element = request.getElement("fields")
            fields_element.appendValue(TICKER_FIELD)
            fields_element.appendValue(BICS_INDUSTRY_FIELD)

            for message in self._response_messages(request):
                if not message.hasElement("securityData"):
                    continue
                for security_data in message.getElement("securityData").values():
                    security = security_data.getElementAsString("security")
                    if security_data.hasElement("sequenceNumber"):
                        sequence_number = security_data.getElementAsInteger(
                            "sequenceNumber"
                        )
                        input_identifier = identifier_batch[sequence_number]
                    else:
                        input_identifier = security
                    if security_data.hasElement("securityError"):
                        print(
                            f"ERROR: Bloomberg could not resolve "
                            f"{input_identifier}: "
                            f"{security_data.getElement('securityError')}",
                            file=sys.stderr,
                        )
                        continue

                    field_data = security_data.getElement("fieldData")
                    ticker = self._field_value(field_data, TICKER_FIELD)
                    if not ticker:
                        print(
                            f"ERROR: Bloomberg returned no ticker for "
                            f"{input_identifier}; skipping.",
                            file=sys.stderr,
                        )
                        continue
                    industry = self._field_value(field_data, BICS_INDUSTRY_FIELD)
                    if not industry:
                        industry = "UNKNOWN"
                        print(
                            f"WARNING: Bloomberg returned no BICS industry for "
                            f"{input_identifier}; storing UNKNOWN.",
                            file=sys.stderr,
                        )
                    resolved.append(
                        SecurityInfo(
                            input_identifier=input_identifier,
                            bloomberg_security=security,
                            ticker=ticker,
                            industry=industry,
                        )
                    )
        return resolved

    def _historical_rows(
        self,
        securities: Sequence[SecurityInfo],
        field_name: str,
        start_date: date,
        end_date: date,
    ) -> Iterator[tuple[SecurityInfo, date, float]]:
        """Yield valid historical values returned for a Bloomberg field."""
        if self.service is None:
            raise RuntimeError("Bloomberg session has not been opened.")
        if not securities:
            return

        for security_batch in _chunks(securities, self.batch_size):
            request = self.service.createRequest("HistoricalDataRequest")
            security_lookup = {
                security.bloomberg_security: security for security in security_batch
            }
            securities_element = request.getElement("securities")
            for security_name in security_lookup:
                securities_element.appendValue(security_name)
            request.getElement("fields").appendValue(field_name)
            request.set("startDate", start_date.strftime("%Y%m%d"))
            request.set("endDate", end_date.strftime("%Y%m%d"))
            request.set("periodicitySelection", "DAILY")
            request.set("nonTradingDayFillOption", "ACTIVE_DAYS_ONLY")

            for message in self._response_messages(request):
                if not message.hasElement("securityData"):
                    continue
                security_data = message.getElement("securityData")
                security_name = security_data.getElementAsString("security")
                security = security_lookup.get(security_name)
                if security is None:
                    print(
                        f"WARNING: Ignoring unexpected Bloomberg security "
                        f"{security_name}.",
                        file=sys.stderr,
                    )
                    continue
                if security_data.hasElement("securityError"):
                    print(
                        f"ERROR: Bloomberg returned an error for "
                        f"{security.input_identifier}: "
                        f"{security_data.getElement('securityError')}",
                        file=sys.stderr,
                    )
                    continue

                for field_data in security_data.getElement("fieldData").values():
                    if not field_data.hasElement(field_name):
                        continue
                    field_element = field_data.getElement(field_name)
                    if field_element.isNull():
                        continue
                    observation_date = field_data.getElementAsDatetime("date").date()
                    try:
                        value = field_element.getValueAsFloat()
                    except (TypeError, ValueError):
                        print(
                            f"WARNING: Non-numeric {field_name} for "
                            f"{security.ticker} on {observation_date}; skipping.",
                            file=sys.stderr,
                        )
                        continue
                    if math.isfinite(value):
                        yield security, observation_date, value

    def get_prices(
        self,
        securities: Sequence[SecurityInfo],
        windows: Iterable[tuple[date, date]],
    ) -> list[PriceRow]:
        """Retrieve daily closing prices for each requested date window."""
        rows: list[PriceRow] = []
        for start_date, end_date in windows:
            for security, observation_date, value in self._historical_rows(
                securities,
                PRICE_FIELD,
                start_date,
                end_date,
            ):
                rows.append(
                    PriceRow(
                        observation_date=observation_date,
                        ticker=security.ticker,
                        industry=security.industry,
                        price=value,
                    )
                )
        return rows

    def get_latest_short_interest(
        self,
        securities: Sequence[SecurityInfo],
        as_of_date: date,
        lookback_days: int,
    ) -> list[ShortInterestRow]:
        """Retrieve each security's latest short-interest value on/before as-of."""
        latest: dict[str, ShortInterestRow] = {}
        recent_start_date = as_of_date - timedelta(days=lookback_days)

        def collect(targets: Sequence[SecurityInfo], start_date: date) -> None:
            for security, observation_date, value in self._historical_rows(
                targets,
                SHORT_INTEREST_FIELD,
                start_date,
                as_of_date,
            ):
                existing = latest.get(security.ticker)
                if existing is None or observation_date > existing.observation_date:
                    latest[security.ticker] = ShortInterestRow(
                        observation_date=observation_date,
                        ticker=security.ticker,
                        industry=security.industry,
                        short_interest_percent_float=value,
                    )

        collect(securities, recent_start_date)
        missing = [
            security for security in securities if security.ticker not in latest
        ]
        if missing and recent_start_date > date(1900, 1, 1):
            collect(missing, date(1900, 1, 1))

        for security in securities:
            if security.ticker not in latest:
                print(
                    f"WARNING: No {SHORT_INTEREST_FIELD} observation found for "
                    f"{security.ticker} through {as_of_date}.",
                    file=sys.stderr,
                )
        return list(latest.values())

    def get_market_caps(
        self,
        securities: Sequence[SecurityInfo],
        as_of_date: date,
    ) -> list[MarketCapRow]:
        """Retrieve the latest market cap on or before the as-of date."""
        latest: dict[str, MarketCapRow] = {}
        start_date = as_of_date - timedelta(days=MARKET_CAP_LOOKBACK_DAYS)
        for security, observation_date, value in self._historical_rows(
            securities,
            MARKET_CAP_FIELD,
            start_date,
            as_of_date,
        ):
            existing = latest.get(security.ticker)
            if existing is None or observation_date > existing.observation_date:
                latest[security.ticker] = MarketCapRow(
                    observation_date=observation_date,
                    ticker=security.ticker,
                    industry=security.industry,
                    market_cap=value,
                )

        for security in securities:
            if security.ticker not in latest:
                print(
                    f"WARNING: No {MARKET_CAP_FIELD} observation found for "
                    f"{security.ticker} in the {MARKET_CAP_LOOKBACK_DAYS} days "
                    f"through {as_of_date}.",
                    file=sys.stderr,
                )
        return list(latest.values())


def initialize_database(connection: sqlite3.Connection) -> None:
    """Create destination tables and indexes when they do not already exist."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS prices (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            industry TEXT NOT NULL,
            price REAL NOT NULL,
            PRIMARY KEY (date, ticker)
        );

        CREATE TABLE IF NOT EXISTS short_interest (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            industry TEXT NOT NULL,
            short_interest_percent_float REAL NOT NULL,
            PRIMARY KEY (date, ticker)
        );

        CREATE TABLE IF NOT EXISTS market_caps (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            industry TEXT NOT NULL,
            market_cap REAL NOT NULL,
            PRIMARY KEY (date, ticker)
        );

        CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
            ON prices (ticker, date);
        CREATE INDEX IF NOT EXISTS idx_short_interest_ticker_date
            ON short_interest (ticker, date);
        CREATE INDEX IF NOT EXISTS idx_market_caps_ticker_date
            ON market_caps (ticker, date);
        """
    )


def store_rows(
    database_path: Path,
    price_rows: Sequence[PriceRow],
    short_interest_rows: Sequence[ShortInterestRow],
    market_cap_rows: Sequence[MarketCapRow] = (),
) -> None:
    """Upsert retrieved Bloomberg observations into SQLite atomically."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            initialize_database(connection)
            connection.executemany(
                """
                INSERT INTO prices (date, ticker, industry, price)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (date, ticker) DO UPDATE SET
                    industry = excluded.industry,
                    price = excluded.price
                """,
                (
                    (
                        row.observation_date.isoformat(),
                        row.ticker,
                        row.industry,
                        row.price,
                    )
                    for row in price_rows
                ),
            )
            connection.executemany(
                """
                INSERT INTO market_caps (date, ticker, industry, market_cap)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (date, ticker) DO UPDATE SET
                    industry = excluded.industry,
                    market_cap = excluded.market_cap
                """,
                (
                    (
                        row.observation_date.isoformat(),
                        row.ticker,
                        row.industry,
                        row.market_cap,
                    )
                    for row in market_cap_rows
                ),
            )
            connection.executemany(
                """
                INSERT INTO short_interest (
                    date,
                    ticker,
                    industry,
                    short_interest_percent_float
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT (date, ticker) DO UPDATE SET
                    industry = excluded.industry,
                    short_interest_percent_float = excluded.short_interest_percent_float
                """,
                (
                    (
                        row.observation_date.isoformat(),
                        row.ticker,
                        row.industry,
                        row.short_interest_percent_float,
                    )
                    for row in short_interest_rows
                ),
            )


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve Bloomberg prices, short interest, and market caps for "
            "tickers in an XLSX workbook and store the results in SQLite."
        )
    )
    parser.add_argument(
        "input_xlsx",
        type=Path,
        help=(
            "estimation_universe_YYYYMMDD.xlsx workbook containing a "
            "Ticker/All column."
        ),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("bloomberg_history.db"),
        help="Destination SQLite database (default: bloomberg_history.db).",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Bloomberg Desktop API host (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8194,
        help="Bloomberg Desktop API port (default: 8194).",
    )
    parser.add_argument(
        "--short-interest-lookback-days",
        type=int,
        default=730,
        help=(
            "Initial days to search backward for short interest; securities with "
            "no result are retried from 1900 (default: 730)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Securities per Bloomberg request (default: 50).",
    )
    parser.add_argument(
        "--ticker-suffix",
        default="US Equity",
        help=(
            "Bloomberg market-sector suffix appended to workbook tickers "
            "(default: 'US Equity')."
        ),
    )
    parsed = parser.parse_args(arguments)
    if parsed.short_interest_lookback_days <= 0:
        parser.error("--short-interest-lookback-days must be positive")
    if not 1 <= parsed.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    if parsed.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the Bloomberg extraction and SQLite load."""
    options = parse_arguments(arguments)
    try:
        workbook_input = read_workbook_input(
            options.input_xlsx,
            ticker_suffix=options.ticker_suffix,
        )
        windows = historical_price_windows(workbook_input.as_of_date)
        print(
            f"Loaded {len(workbook_input.identifiers)} securities; data date "
            f"{workbook_input.as_of_date}."
        )

        with BloombergClient(
            host=options.host,
            port=options.port,
            batch_size=options.batch_size,
        ) as client:
            securities = client.resolve_securities(workbook_input.identifiers)
            if not securities:
                raise RuntimeError("Bloomberg could not resolve any input securities.")
            print(
                f"Resolved {len(securities)} of "
                f"{len(workbook_input.identifiers)} securities."
            )
            prices = client.get_prices(securities, windows)
            short_interest = client.get_latest_short_interest(
                securities,
                workbook_input.as_of_date,
                options.short_interest_lookback_days,
            )
            market_caps = client.get_market_caps(
                securities,
                workbook_input.as_of_date,
            )

        store_rows(options.database, prices, short_interest, market_caps)
        print(
            f"Stored {len(prices)} price rows and {len(short_interest)} "
            f"short-interest rows and {len(market_caps)} market-cap rows in "
            f"{options.database}."
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, sqlite3.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
