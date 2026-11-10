"""Record live Binance bookTicker (spot and USD-M perp) to Parquet.

Why: data.binance.vision publishes no spot bookTicker, and its perp bookTicker ends on
2024-03-30. Recording both streams in one process gives top-of-book quotes for both
venues on the same clock.

Timestamps: spot bookTicker messages carry no exchange timestamp, so every row (both
venues) gets `local_time`, the moment we received it. That is the common clock for
cross-venue work. Perp rows also keep the exchange times `transaction_time` and
`event_time`; spot rows have nulls there.

Output, one Parquet file per flush, partitioned by UTC day of local_time:
    data/live/<venue>_bookTicker/<YYYY-MM-DD>/part-<first local_time in us>.parquet
Read a day with pl.scan_parquet("data/live/spot_bookTicker/2026-11-11/*.parquet").

`session` counts connections per venue. A new session means we reconnected, so the
time between sessions is a gap with no data.

Usage:
    python -m crypto_microstructure.record --hours 24 [--symbol BTCUSDT]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import websockets

from crypto_microstructure.download import DATA_DIR

STREAM_URLS = {
    "spot": "wss://stream.binance.com:9443/ws/{symbol}@bookTicker",
    "perp": "wss://fstream.binance.com/ws/{symbol}@bookTicker",
}

# Raw row layout produced by parse_message (epoch integers, converted on write).
RAW_SCHEMA = {
    "update_id": pl.Int64,
    "bid_price": pl.Float64,
    "bid_qty": pl.Float64,
    "ask_price": pl.Float64,
    "ask_qty": pl.Float64,
    "transaction_time": pl.Int64,  # ms, perp only
    "event_time": pl.Int64,  # ms, perp only
    "local_time": pl.Int64,  # us
    "session": pl.Int32,
}

Row = tuple


def parse_message(msg: dict, local_us: int, session: int) -> Row:
    """One bookTicker message -> raw row. Prices arrive as strings."""
    return (
        msg["u"],
        float(msg["b"]),
        float(msg["B"]),
        float(msg["a"]),
        float(msg["A"]),
        msg.get("T"),
        msg.get("E"),
        local_us,
        session,
    )


def rows_to_frame(rows: list[Row]) -> pl.DataFrame:
    """Raw rows -> typed frame with the same column names as the historical files."""
    return pl.DataFrame(rows, schema=RAW_SCHEMA, orient="row").with_columns(
        pl.from_epoch("transaction_time", time_unit="ms").dt.replace_time_zone("UTC"),
        pl.from_epoch("event_time", time_unit="ms").dt.replace_time_zone("UTC"),
        pl.from_epoch("local_time", time_unit="us").dt.replace_time_zone("UTC"),
    )


def utc_day(local_us: int) -> str:
    return datetime.fromtimestamp(local_us / 1e6, tz=timezone.utc).date().isoformat()


class ParquetBuffer:
    """Collects rows in memory and writes them out as Parquet part files.

    Rows are only appended, so a crash loses at most the unflushed rows. A buffer never
    spans two UTC days: when a row from a new day arrives, the old day is flushed first.
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.rows: list[Row] = []
        self.day: str | None = None

    def add(self, row: Row) -> None:
        day = utc_day(row[7])
        if self.day is not None and day != self.day:
            self.flush()
        self.day = day
        self.rows.append(row)

    def flush(self) -> Path | None:
        if not self.rows:
            return None
        path = self.out_dir / self.day / f"part-{self.rows[0][7]}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows_to_frame(self.rows).write_parquet(path)
        self.rows = []
        return path


async def record_stream(venue: str, url: str, buffer: ParquetBuffer, stop_at: float) -> None:
    """Receive one stream until stop_at, reconnecting with exponential backoff.

    Binance closes connections after 24h and on maintenance; we reconnect and bump
    the session number. The websockets library answers the server's pings for us.
    """
    session, backoff = 0, 1.0
    while time.time() < stop_at:
        try:
            async with websockets.connect(url) as ws:
                session += 1
                backoff = 1.0
                print(f"{venue}: connected, session {session}", flush=True)
                async for raw in ws:
                    local_us = time.time_ns() // 1000  # stamp before parsing
                    buffer.add(parse_message(json.loads(raw), local_us, session))
                    if local_us / 1e6 >= stop_at:
                        break
        except (websockets.ConnectionClosed, OSError) as e:
            print(f"{venue}: disconnected ({e!r}), retry in {backoff:.0f}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


async def flush_periodically(buffers: dict[str, ParquetBuffer], every_s: float) -> None:
    while True:
        await asyncio.sleep(every_s)
        for venue, buf in buffers.items():
            n = len(buf.rows)
            if buf.flush():
                print(f"{venue}: flushed {n:,} rows", flush=True)


async def record(symbol: str, hours: float, flush_every_s: float, data_dir: Path) -> None:
    stop_at = time.time() + hours * 3600
    buffers = {
        venue: ParquetBuffer(data_dir / "live" / f"{venue}_bookTicker") for venue in STREAM_URLS
    }
    flusher = asyncio.create_task(flush_periodically(buffers, flush_every_s))
    try:
        await asyncio.gather(
            *(
                record_stream(venue, url.format(symbol=symbol.lower()), buffers[venue], stop_at)
                for venue, url in STREAM_URLS.items()
            )
        )
    finally:
        flusher.cancel()
        for buf in buffers.values():
            buf.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--flush-every", type=float, default=60.0, help="seconds")
    args = parser.parse_args()
    try:
        asyncio.run(record(args.symbol, args.hours, args.flush_every, DATA_DIR))
    except KeyboardInterrupt:
        print("stopped (buffers flushed)")


if __name__ == "__main__":
    main()
