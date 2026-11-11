"""Effective spread per taker event by size, and maker markouts by position vs the touch.

Usage:
    python examples/perp_taker_events.py --date 2024-03-01 --start 14:00 --end 15:00
"""

import argparse
from datetime import date, datetime, time, timedelta, timezone

import polars as pl

from crypto_microstructure.load import TICK, load_perp
from crypto_microstructure.metrics import (
    attach_mid,
    drop_trades_near_end,
    effective_spread_bps,
    group_taker_events,
    markouts_bps,
    trade_sign,
)

HORIZONS = ["1s", "5s", "30s"]
MAX_HORIZON = timedelta(seconds=30)
PAD = timedelta(minutes=1)  # quotes before the window, so the first trades have a mid
BUCKETS = ["inside touch", "at touch", "1 through", "2-10 through", ">10 through"]


def attach_touch(trades: pl.LazyFrame, quotes: pl.LazyFrame) -> pl.LazyFrame:
    """Bid/ask of the last quote strictly before each trade (same rule as attach_mid)."""
    q = quotes.select(pl.col("time").alias("_qtime"), "bid", "ask").sort("_qtime")
    t = trades.with_columns(pl.col("time").dt.offset_by("-1us").alias("_key")).sort("_key")
    return t.join_asof(q, left_on="_key", right_on="_qtime", strategy="backward").drop(
        "_key", "_qtime"
    )


def vw(col: str, w: pl.Expr) -> pl.Expr:
    return ((pl.col(col) * w).sum() / w.sum()).alias(col)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, default=date(2024, 3, 1))
    parser.add_argument("--start", type=time.fromisoformat, default=time(14, 0))
    parser.add_argument("--end", type=time.fromisoformat, default=time(15, 0))
    args = parser.parse_args()
    start = datetime.combine(args.date, args.start, tzinfo=timezone.utc)
    end = datetime.combine(args.date, args.end, tzinfo=timezone.utc)

    day = load_perp(args.date)
    trades = day.trades.filter(pl.col("time").is_between(start, end, closed="left"))
    quotes = day.quotes.filter(
        pl.col("time").is_between(start - PAD, end + MAX_HORIZON + PAD)
    )
    trades = drop_trades_near_end(trades, quotes, HORIZONS[-1])

    # --- 2. ES per taker event (VWAP vs mid strictly before the event), by size quintile
    events = effective_spread_bps(
        attach_mid(group_taker_events(trades), quotes).with_columns(price=pl.col("vwap"))
    ).collect()
    total_notional = events["notional"].sum()
    by_size = (
        events.with_columns(
            quintile=pl.col("notional").qcut(5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        )
        .group_by("quintile")
        .agg(
            n_events=pl.len(),
            min_notional=pl.col("notional").min(),
            max_notional=pl.col("notional").max(),
            notional_share=pl.col("notional").sum() / total_notional,
            mean_es_bps=pl.col("es_bps").mean(),
            vw_es_bps=(pl.col("es_bps") * pl.col("notional")).sum() / pl.col("notional").sum(),
            mean_levels=pl.col("n_levels").mean(),
            mean_ticks_walked=pl.col("ticks_walked").mean(),
        )
        .sort("quintile")
    )

    # --- 3. Maker markouts per aggTrade, notional-weighted, by position vs the touch
    s = trade_sign(pl.col("is_buyer_maker"))
    touch = pl.when(s == 1).then(pl.col("ask")).otherwise(pl.col("bid"))
    ticks_through = (s * (pl.col("price") - touch) / TICK).round(0)
    bucket = (
        pl.when(ticks_through < 0).then(pl.lit(BUCKETS[0]))
        .when(ticks_through == 0).then(pl.lit(BUCKETS[1]))
        .when(ticks_through == 1).then(pl.lit(BUCKETS[2]))
        .when(ticks_through <= 10).then(pl.lit(BUCKETS[3]))
        .otherwise(pl.lit(BUCKETS[4]))
        .cast(pl.Enum(BUCKETS))
    )
    per_trade = (
        effective_spread_bps(markouts_bps(attach_touch(trades, quotes), quotes, HORIZONS))
        .with_columns(bucket=bucket, notional=pl.col("price") * pl.col("qty"))
        .collect()
    )
    w = pl.col("notional")
    by_touch = (
        per_trade.group_by("bucket")
        .agg(
            n_aggtrades=pl.len(),
            notional_share=w.sum() / per_trade["notional"].sum(),
            vw_es_bps=vw("es_bps", w),
            **{f"vw_markout_{h}": vw(f"markout_{h}_bps", w) for h in HORIZONS},
        )
        .sort("bucket")
    )
    all_row = per_trade.select(
        bucket=pl.lit("all"),
        n_aggtrades=pl.len(),
        notional_share=pl.lit(1.0),
        vw_es_bps=vw("es_bps", w),
        **{f"vw_markout_{h}": vw(f"markout_{h}_bps", w) for h in HORIZONS},
    )

    print(
        f"window {start:%Y-%m-%d %H:%M}–{end:%H:%M} UTC | aggTrades: {per_trade.height} | "
        f"taker events: {events.height} | events without a mid: {events['mid'].null_count()}"
    )
    cfg = dict(tbl_cols=-1, tbl_width_chars=200, float_precision=3, tbl_hide_dataframe_shape=True)
    with pl.Config(**cfg):
        print("\nES per taker event by notional quintile (USDT, bps)")
        print(by_size)
        print("\nMaker markouts per aggTrade by position vs touch (notional-weighted, bps)")
        print(pl.concat([by_touch.with_columns(pl.col("bucket").cast(pl.String)), all_row]))


if __name__ == "__main__":
    main()
