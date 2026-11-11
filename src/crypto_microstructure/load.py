"""Load one day of perp data from Parquet with the column names metrics.py expects."""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

import polars as pl

from crypto_microstructure.download import DATA_DIR

# USD-M perp price ticks in March 2024, inferred as the smallest positive spread in the
# 2024-03-01 bookTicker files (all quotes of the five alts lie on that grid; BTCUSDT has a
# few hundred off-grid quotes). Binance changes ticks from time to time, so check them
# again before using other dates.
TICKS: dict[str, float] = {
    "BTCUSDT": 0.1,
    "XRPUSDT": 0.0001,
    "ADAUSDT": 0.0001,
    "DOGEUSDT": 0.00001,
    "LTCUSDT": 0.01,
    "LINKUSDT": 0.001,
    "AVAXUSDT": 0.001,
}
TICK = TICKS["BTCUSDT"]  # default for code written for BTCUSDT only


class PerpDay(NamedTuple):
    quotes: pl.LazyFrame  # time, bid, ask, update_id
    trades: pl.LazyFrame  # time, price, qty, is_buyer_maker, first/last_trade_id, off_grid
    n_off_grid_dropped: int
    tick: float  # price tick used for the grid checks


def on_grid(col: str, tick: float = TICK) -> pl.Expr:
    """True if `col` is a multiple of `tick` (within float rounding).

    The tolerance is in units of ticks, so it works the same for 0.1 and 0.00001.
    """
    x = pl.col(col) / tick
    return (x - x.round()).abs() < 1e-6


def load_perp(
    day: date,
    symbol: str = "BTCUSDT",
    tick: float | None = None,
    data_dir=DATA_DIR,
) -> PerpDay:
    """Perp bookTicker and aggTrades of `symbol` for `day`, renamed for metrics.py.

    tick: price tick; defaults to TICKS[symbol]. Pass it for symbols not in TICKS.

    Quotes with a bid or ask off the tick grid are dropped. The as-of join then keeps
    the previous valid quote live until the next one.
    Off-tick trades are kept and flagged with `off_grid` for robustness checks.
    """
    if tick is None:
        if symbol not in TICKS:
            raise ValueError(f"no tick known for {symbol}; pass tick=")
        tick = TICKS[symbol]

    root = data_dir / "parquet"
    book = pl.scan_parquet(root / "perp_bookTicker" / f"{symbol}-bookTicker-{day}.parquet")
    agg = pl.scan_parquet(root / "perp_aggTrades" / f"{symbol}-aggTrades-{day}.parquet")

    valid = on_grid("bid_price", tick) & on_grid("ask_price", tick)
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
        (~on_grid("price", tick)).alias("off_grid"),
    )
    return PerpDay(quotes, trades, n_dropped, tick)
