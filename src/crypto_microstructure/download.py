"""Download Binance public market data from data.binance.vision and convert it to Parquet.

Layout on disk:
    data/raw/<dataset>/<file>.zip         original archive (checksum verified)
    data/parquet/<dataset>/<file>.parquet  typed columns, timestamps as UTC datetimes

Availability notes (checked against the bucket listing, Nov 2026):
- Spot has no bookTicker dataset at all, only aggTrades, klines and trades.
- USD-M futures bookTicker daily files exist only from 2023-05-16 to 2024-03-30.
- fundingRate is published only as monthly files.

Usage:
    python -m crypto_microstructure.download --date 2024-03-12 [--symbol BTCUSDT]
"""

from __future__ import annotations

import argparse
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
    remote_dir: str  # e.g. "futures/um/daily/bookTicker"
    columns: dict[str, type[pl.DataType]]
    time_cols: tuple[str, ...]
    monthly: bool = False


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
}

# Not downloadable: listed only so check_availability() documents the answer.
SPOT_BOOKTICKER_DIR = "spot/daily/bookTicker"


def file_name(dataset: Dataset, symbol: str, day: date) -> str:
    period = day.strftime("%Y-%m") if dataset.monthly else day.isoformat()
    return f"{symbol}-{dataset.name}-{period}.zip"


def file_url(dataset: Dataset, symbol: str, day: date) -> str:
    return f"{BASE_URL}/{dataset.remote_dir}/{symbol}/{file_name(dataset, symbol, day)}"


def url_exists(url: str) -> bool:
    """HEAD request: True on 200, False on 404."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
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


def download(url: str, dest: Path) -> Path:
    """Download url to dest (skipped if present) and verify the published SHA-256."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)
    with urllib.request.urlopen(url + ".CHECKSUM", timeout=30) as resp:
        expected = resp.read().decode().split()[0]
    actual = _sha256(tmp)
    if actual != expected:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {url}: {actual} != {expected}")
    tmp.rename(dest)
    return dest


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
    name = file_name(ds, symbol, day)
    out = data_dir / "parquet" / key / name.replace(".zip", ".parquet")
    if out.exists():
        return out
    zip_path = download(file_url(ds, symbol, day), data_dir / "raw" / key / name)
    with tempfile.TemporaryDirectory(dir=data_dir) as tmp, zipfile.ZipFile(zip_path) as zf:
        (member,) = zf.namelist()  # each archive holds exactly one CSV
        csv_path = Path(zf.extract(member, tmp))
        csv_to_parquet(csv_path, ds, out)
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
