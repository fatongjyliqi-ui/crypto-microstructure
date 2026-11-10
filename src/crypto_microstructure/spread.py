"""Mid price and quoted spread from top-of-book (bookTicker) data."""

from __future__ import annotations

from datetime import datetime

import polars as pl

BPS = 1e4


def add_mid_and_quoted_spread(book: pl.LazyFrame) -> pl.LazyFrame:
    """Add mid, quoted spread in price units and in bps of mid.

    mid        = (bid + ask) / 2
    spread     = ask - bid
    spread_bps = spread / mid * 10_000
    """
    mid = (pl.col("bid_price") + pl.col("ask_price")) / 2
    spread = pl.col("ask_price") - pl.col("bid_price")
    return book.with_columns(
        mid=mid,
        spread=spread,
        spread_bps=spread / mid * BPS,
    )


def sample_on_grid(
    book: pl.LazyFrame,
    start: datetime,
    end: datetime,
    every: str = "1s",
    time_col: str = "transaction_time",
) -> pl.LazyFrame:
    """Book state at each point of a regular clock grid.

    For grid time t we take the last quote with time <= t (an as-of join). bookTicker is
    event-driven: averaging raw updates over-weights busy moments, whereas averaging
    the grid samples weights each quote by how long it was live (time-weighted).
    Grid points before the first quote get nulls.
    """
    grid = pl.LazyFrame(
        {"time": pl.datetime_range(start, end, every, eager=True, time_zone="UTC")}
    )
    quotes = book.rename({time_col: "time"}).sort("time")
    return grid.join_asof(quotes, on="time", strategy="backward")
