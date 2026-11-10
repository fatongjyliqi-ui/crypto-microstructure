"""Download Binance market data and convert it to Parquet.

Sources:
- data.binance.vision: perp bookTicker, aggTrades, fundingRate; spot trades.
- Tardis.dev free tier: spot quotes (best bid/ask). Free only for the first day of each
  month, without an API key.

Layout on disk:
    data/raw/<dataset>/<file>.zip|.csv.gz  original archive (Binance: checksum verified)
    data/parquet/<dataset>/<file>.parquet  typed columns, timestamps as UTC datetimes

Availability notes (checked Nov 2026):
- data.binance.vision has no spot bookTicker at all, only aggTrades, klines and trades.
- USD-M futures bookTicker daily files exist only from 2023-05-16 to 2024-03-30.
- fundingRate is published only as monthly files.
- Tardis spot quotes are 100 ms snapshots rebuilt from Binance's depth@100ms stream, not
  every top-of-book change. `timestamp` is the exchange event time (ms precision),
  `local_timestamp` is when Tardis received it (median ~4 ms later).

Usage:
    python -m crypto_microstructure.download --date 2024-03-12 [--symbol BTCUSDT]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

BASE_URL = "https://data.binance.vision/data"
TARDIS_URL = "https://datasets.tardis.dev/v1"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Dataset:
    """One file type on data.binance.vision.

    columns: canonical column names and dtypes, in file order. Some files have a header
        row and some do not, so we always read positionally and apply these names.
    time_cols: integer epoch columns to convert to datetimes. Binance switched spot
        timestamps from milliseconds to microseconds on 2025-01-01, so the unit is
        inferred per file (see _infer_time_unit).
    """

    name: str
    remote_dir: str  # binance: "futures/um/daily/bookTicker"; tardis: "binance/quotes"
    columns: dict[str, type[pl.DataType]]
    time_cols: tuple[str, ...]
    monthly: bool = False
    source: str = "binance"  # "binance" (data.binance.vision) or "tardis"


DATASETS: dict[str, Dataset] = {
    "perp_bookTicker": Dataset(
        name="bookTicker",
        remote_dir="futures/um/daily/bookTicker",
        columns={
            "update_id": pl.Int64,
            "bid_price": pl.Float64,
            "bid_qty": pl.Float64,
            "ask_price": pl.Float64,
            "ask_qty": pl.Float64,
            "transaction_time": pl.Int64,
            "event_time": pl.Int64,
        },
        time_cols=("transaction_time", "event_time"),
    ),
    "perp_aggTrades": Dataset(
        name="aggTrades",
        remote_dir="futures/um/daily/aggTrades",
        columns={
            "agg_trade_id": pl.Int64,
            "price": pl.Float64,
            "qty": pl.Float64,
            "first_trade_id": pl.Int64,
            "last_trade_id": pl.Int64,
            "transact_time": pl.Int64,
            "is_buyer_maker": pl.Boolean,
        },
        time_cols=("transact_time",),
    ),
    "perp_fundingRate": Dataset(
        name="fundingRate",
        remote_dir="futures/um/monthly/fundingRate",
        columns={
            "calc_time": pl.Int64,
            "funding_interval_hours": pl.Int64,
            "funding_rate": pl.Float64,
        },
        time_cols=("calc_time",),
        monthly=True,
    ),
    "spot_trades": Dataset(
        name="trades",
        remote_dir="spot/daily/trades",
        columns={
            "trade_id": pl.Int64,
            "price": pl.Float64,
            "qty": pl.Float64,
            "quote_qty": pl.Float64,
            "time": pl.Int64,
            "is_buyer_maker": pl.Boolean,
            "is_best_match": pl.Boolean,
        },
        time_cols=("time",),
    ),
    "spot_quotes": Dataset(
        name="quotes",
        remote_dir="binance/quotes",
        columns={
            "exchange": pl.String,
            "symbol": pl.String,
            "timestamp": pl.Int64,
            "local_timestamp": pl.Int64,
            "ask_qty": pl.Float64,
            "ask_price": pl.Float64,
            "bid_price": pl.Float64,
            "bid_qty": pl.Float64,
        },
        time_cols=("timestamp", "local_timestamp"),
        source="tardis",
    ),
}

# Not downloadable: listed only so check_availability() documents the answer.
SPOT_BOOKTICKER_DIR = "spot/daily/bookTicker"


def file_name(dataset: Dataset, symbol: str, day: date) -> str:
    """Local file name; for Binance it is also the remote name."""
    if dataset.source == "tardis":
        return f"{symbol}-{dataset.name}-{day.isoformat()}.csv.gz"
    period = day.strftime("%Y-%m") if dataset.monthly else day.isoformat()
    return f"{symbol}-{dataset.name}-{period}.zip"


def file_url(dataset: Dataset, symbol: str, day: date) -> str:
    if dataset.source == "tardis":
        return f"{TARDIS_URL}/{dataset.remote_dir}/{day:%Y/%m/%d}/{symbol}.csv.gz"
    return f"{BASE_URL}/{dataset.remote_dir}/{symbol}/{file_name(dataset, symbol, day)}"


def url_exists(url: str) -> bool:
    """Request the first byte: 200/206 -> True; 401/403/404 -> False.

    A ranged GET instead of HEAD because Tardis does not answer HEAD like a file.
    Tardis returns 401 for days outside the free tier.
    """
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status in (200, 206)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            return False
        raise


def check_availability(symbol: str, day: date) -> dict[str, bool]:
    """Which files exist for this symbol and day (monthly files: the day's month)."""
    result = {key: url_exists(file_url(ds, symbol, day)) for key, ds in DATASETS.items()}
    spot_book = f"{BASE_URL}/{SPOT_BOOKTICKER_DIR}/{symbol}/{symbol}-bookTicker-{day}.zip"
    result["spot_bookTicker"] = url_exists(spot_book)
    return result


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, verify_checksum: bool = True) -> Path:
    """Download url to dest (skipped if present).

    Binance publishes a SHA-256 next to each file, which we verify. Tardis publishes none.
    Either way we write to a .part file and rename at the end, so an interrupted
    download never looks complete.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)
    if verify_checksum:
        with urllib.request.urlopen(url + ".CHECKSUM", timeout=30) as resp:
            expected = resp.read().decode().split()[0]
        actual = _sha256(tmp)
        if actual != expected:
            tmp.unlink()
            raise ValueError(f"checksum mismatch for {url}: {actual} != {expected}")
    tmp.rename(dest)
    return dest


def extract_csv(archive: Path, out_dir: Path) -> Path:
    """Unpack a Binance .zip (exactly one CSV inside) or a Tardis .csv.gz into out_dir."""
    if archive.name.endswith(".csv.gz"):
        csv_path = out_dir / archive.name.removesuffix(".gz")
        with gzip.open(archive, "rb") as src, csv_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        return csv_path
    with zipfile.ZipFile(archive) as zf:
        (member,) = zf.namelist()
        return Path(zf.extract(member, out_dir))


def _has_header(csv_path: Path) -> bool:
    """Newer futures files start with a header row, spot files do not."""
    with csv_path.open() as f:
        first = f.readline()
    return not first[:1].isdigit()


def _infer_time_unit(csv_path: Path, has_header: bool, col_idx: int) -> str:
    """Epoch in ms has 13 digits until year 2286; in us it has 16."""
    with csv_path.open() as f:
        if has_header:
            f.readline()
        value = int(f.readline().split(",")[col_idx])
    return "us" if value > 10**14 else "ms"


def csv_to_parquet(csv_path: Path, dataset: Dataset, out_path: Path) -> Path:
    """Stream a Binance CSV into Parquet with canonical names and UTC datetimes."""
    has_header = _has_header(csv_path)
    names = list(dataset.columns)
    lf = pl.scan_csv(
        csv_path,
        has_header=False,
        skip_rows=1 if has_header else 0,
        new_columns=names,
        schema_overrides=dataset.columns,
    )
    time_exprs = []
    for col in dataset.time_cols:
        unit = _infer_time_unit(csv_path, has_header, names.index(col))
        time_exprs.append(pl.from_epoch(col, time_unit=unit).dt.replace_time_zone("UTC"))
    lf = lf.with_columns(time_exprs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(out_path)
    return out_path


def fetch(key: str, symbol: str, day: date, data_dir: Path = DATA_DIR) -> Path:
    """Download one dataset file and convert it; returns the Parquet path."""
    ds = DATASETS[key]
    if ds.source == "tardis" and day.day != 1:
        raise ValueError(f"{key}: Tardis free tier covers only the 1st of each month, got {day}")
    name = file_name(ds, symbol, day)
    out = data_dir / "parquet" / key / (name.split(".")[0] + ".parquet")
    if out.exists():
        return out
    archive = download(
        file_url(ds, symbol, day),
        data_dir / "raw" / key / name,
        verify_checksum=ds.source == "binance",
    )
    with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
        csv_to_parquet(extract_csv(archive, Path(tmp)), ds, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()

    available = check_availability(args.symbol, args.date)
    for key, ok in available.items():
        print(f"{key:18s} {'available' if ok else 'MISSING'}")
    for key in DATASETS:
        if available[key]:
            path = fetch(key, args.symbol, args.date)
            n_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
            print(f"wrote {path} ({n_rows:,} rows)")


if __name__ == "__main__":
    main()
