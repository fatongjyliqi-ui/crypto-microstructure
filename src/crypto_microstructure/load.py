"""Load one day of perp data from Parquet with the column names metrics.py expects."""

from __future__ import annotations

import math
import warnings
from datetime import date
from typing import NamedTuple

import numpy as np
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
PRICE_DECIMALS = 8  # Binance prices have at most 8 decimals
MAX_OFF_GRID_SHARE = 0.01  # above this share of off-grid quotes, the tick is suspect


class PerpDay(NamedTuple):
    quotes: pl.LazyFrame  # time, bid, ask, update_id
    trades: pl.LazyFrame  # time, price, qty, is_buyer_maker, first/last_trade_id, off_grid
    n_off_grid_dropped: int
    tick: float  # price tick used for the grid checks
    observed_tick: float  # gcd of the gaps between distinct on-grid quote prices


def on_grid(col: str, tick: float) -> pl.Expr:
    """True if `col` is a multiple of `tick` (within float rounding).

    The tolerance is in units of ticks, so it works the same for 0.1 and 0.00001.
    """
    x = pl.col(col) / tick
    return (x - x.round()).abs() < 1e-6


def observed_tick(prices: pl.Series, decimals: int = PRICE_DECIMALS) -> float:
    """Finest grid the prices actually use: gcd of the gaps between distinct prices.

    Prices are mapped to integers k = round(p * 10**decimals). For distinct sorted
    k_1 < ... < k_n, the gcd of (k_2 - k_1, ..., k_n - k_{n-1}) is the largest g such that
    every price lies on {k_1 + j * g}. Returns g / 10**decimals, or NaN for fewer than two
    distinct prices.

    Given the tick, a correct value equals it; a larger value means the data only use
    multiples of the assumed tick (the assumed tick may be too small).
    """
    scale = 10**decimals
    k = np.unique(np.rint(prices.drop_nulls().to_numpy() * scale).astype(np.int64))
    if len(k) < 2:
        return math.nan
    return float(np.gcd.reduce(np.diff(k))) / scale


def check_tick(
    tick: float, observed: float, n_off_grid: int, n_quotes: int, label: str
) -> None:
    """Warn if the assumed tick looks too small (observed grid coarser) or too large
    (many quotes off its grid)."""
    if not math.isnan(observed) and observed > tick * (1 + 1e-9):
        warnings.warn(
            f"{label}: on-grid quotes only use multiples of {observed:g}, "
            f"larger than the assumed tick {tick:g}; the tick may be too small",
            stacklevel=3,
        )
    if n_quotes and n_off_grid / n_quotes > MAX_OFF_GRID_SHARE:
        warnings.warn(
            f"{label}: {n_off_grid} of {n_quotes} quotes ({n_off_grid / n_quotes:.1%}) are "
            f"off the {tick:g} grid; the tick may be too large",
            stacklevel=3,
        )


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
    Warns if the tick looks wrong (see `check_tick`); the observed grid is returned.
    """
    if tick is None:
        if symbol not in TICKS:
            raise ValueError(f"no tick known for {symbol}; pass tick=")
        tick = TICKS[symbol]

    root = data_dir / "parquet"
    book = pl.scan_parquet(root / "perp_bookTicker" / f"{symbol}-bookTicker-{day}.parquet")
    agg = pl.scan_parquet(root / "perp_aggTrades" / f"{symbol}-aggTrades-{day}.parquet")

    valid = on_grid("bid_price", tick) & on_grid("ask_price", tick)
    counts = book.select(n=pl.len(), n_off=(~valid).sum()).collect()
    n_quotes, n_dropped = counts["n"].item(), counts["n_off"].item()
    levels = (
        book.filter(valid)
        .select(pl.concat_list("bid_price", "ask_price").explode().unique().alias("p"))
        .collect()["p"]
    )
    observed = observed_tick(levels)
    check_tick(tick, observed, n_dropped, n_quotes, f"{symbol} {day}")

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
    return PerpDay(quotes, trades, n_dropped, tick, observed)
