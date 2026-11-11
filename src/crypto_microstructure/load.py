"""Load one day of perp data from Parquet with the column names metrics.py expects."""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

import polars as pl

from crypto_microstructure.download import DATA_DIR

TICK = 0.1  # BTCUSDT perp price tick


class PerpDay(NamedTuple):
    quotes: pl.LazyFrame  # time, bid, ask, update_id
    trades: pl.LazyFrame  # time, price, qty, is_buyer_maker, first/last_trade_id, off_grid
    n_off_grid_dropped: int


def on_grid(col: str, tick: float = TICK) -> pl.Expr:
    """True if `col` is a multiple of `tick` (within float rounding)."""
    x = pl.col(col) / tick
    return (x - x.round()).abs() < 1e-6


def load_perp(day: date, data_dir=DATA_DIR) -> PerpDay:
    """Perp bookTicker and aggTrades for `day`, renamed for metrics.py.

    Quotes with a bid or ask off the 0.1 tick grid are dropped. The as-of join then
    keeps the previous valid quote live until the next one.
    Off-tick trades are kept and flagged with `off_grid` for robustness checks.
    """
    root = data_dir / "parquet"
    book = pl.scan_parquet(root / "perp_bookTicker" / f"BTCUSDT-bookTicker-{day}.parquet")
    agg = pl.scan_parquet(root / "perp_aggTrades" / f"BTCUSDT-aggTrades-{day}.parquet")

    valid = on_grid("bid_price") & on_grid("ask_price")
    n_dropped = book.select((~valid).sum()).collect().item()

    quotes = book.filter(valid).select(
        pl.col("transaction_time").alias("time"),
        pl.col("bid_price").alias("bid"),
        pl.col("ask_price").alias("ask"),
        "update_id",  # orders quotes that share a timestamp
    )
    trades = agg.select(
        pl.col("transact_time").alias("time"),
        "price",
        "qty",
        "is_buyer_maker",
        "first_trade_id",
        "last_trade_id",
        (~on_grid("price")).alias("off_grid"),
    )
    return PerpDay(quotes, trades, n_dropped)
