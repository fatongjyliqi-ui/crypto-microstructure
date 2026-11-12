"""Taker-event ES by size and maker markouts by position vs the touch, with bootstrap CIs.

Usage:
    python examples/perp_taker_events.py --date 2024-03-01                 # full UTC day
    python examples/perp_taker_events.py --symbol XRPUSDT --date 2024-03-01  # other perp
    python examples/perp_taker_events.py --date 2024-03-01 --start 14:00 --end 15:00
    python examples/perp_taker_events.py --date 2024-03-01 --block 1m     # bootstrap blocks

Writes the markout curves (all horizons, by bucket, with CIs) to
data/results/markouts_<symbol>_<date>_<block>.parquet for plotting.
"""

import argparse
from datetime import date, datetime, time, timedelta, timezone

import polars as pl

from crypto_microstructure.download import DATA_DIR
from crypto_microstructure.load import load_perp
from crypto_microstructure.metrics import (
    attach_mid,
    drop_trades_near_end,
    effective_spread_bps,
    group_taker_events,
    markouts_bps,
    trade_sign,
)
from crypto_microstructure.stats import block_bootstrap_wmean

HORIZONS = ["100ms", "500ms", "1s", "2s", "5s", "10s", "30s", "60s"]
TABLE_HORIZONS = ["1s", "5s", "30s"]
MAX_HORIZON = timedelta(seconds=60)
PAD = timedelta(minutes=1)  # quotes before the window, so the first trades have a mid
BUCKETS = ["inside touch", "at touch", "1 through", "2-10 through", ">10 through"]
MERGED = "1-10 through"  # extra group in the saved curves (for the plot), not in the tables
N_BOOT = 1000
SEED = 0
RESULTS_DIR = DATA_DIR / "results"


def horizon_seconds(h: str) -> float:
    return float(h[:-2]) / 1000 if h.endswith("ms") else float(h[:-1])


def attach_touch(trades: pl.LazyFrame, quotes: pl.LazyFrame) -> pl.LazyFrame:
    """Bid/ask of the last quote strictly before each trade (same rule as attach_mid,
    including ties at one timestamp ordered by update_id)."""
    q = quotes.select(pl.col("time").alias("_qtime"), "bid", "ask", "update_id")
    q = q.sort(["_qtime", "update_id"]).drop("update_id")
    t = trades.with_columns(pl.col("time").dt.offset_by("-1us").alias("_key")).sort("_key")
    return t.join_asof(q, left_on="_key", right_on="_qtime", strategy="backward").drop(
        "_key", "_qtime"
    )


def position_bucket(tick: float) -> pl.Expr:
    """Ticks through the touch (taker side) at the last quote before the trade."""
    s = trade_sign(pl.col("is_buyer_maker"))
    touch = pl.when(s == 1).then(pl.col("ask")).otherwise(pl.col("bid"))
    k = (s * (pl.col("price") - touch) / tick).round(0)
    return (
        pl.when(k < 0).then(pl.lit(BUCKETS[0]))
        .when(k == 0).then(pl.lit(BUCKETS[1]))
        .when(k == 1).then(pl.lit(BUCKETS[2]))
        .when(k <= 10).then(pl.lit(BUCKETS[3]))
        .otherwise(pl.lit(BUCKETS[4]))
    )


def vw(col: str, w: str = "notional") -> pl.Expr:
    return (pl.col(col) * pl.col(w)).sum() / pl.col(w).sum()


# ---------------------------------------------------------------- tables


def event_table(ev: pl.DataFrame) -> pl.DataFrame:
    total = ev["notional"].sum()
    return (
        ev.group_by("quintile")
        .agg(
            n_events=pl.len(),
            min_notional=pl.col("notional").min(),
            max_notional=pl.col("notional").max(),
            notional_share=pl.col("notional").sum() / total,
            mean_es_bps=pl.col("es_bps").mean(),
            vw_es_bps=vw("es_bps"),
            mean_levels=pl.col("n_levels").mean(),
            mean_ticks_walked=pl.col("ticks_walked").mean(),
        )
        .sort("quintile")
    )


def markout_table(pt: pl.DataFrame, horizons: list[str]) -> pl.DataFrame:
    """Notional-weighted ES and markouts per bucket, plus an 'all' row."""
    total = pt["notional"].sum()
    aggs = dict(
        n_aggtrades=pl.len(),
        notional_share=pl.col("notional").sum() / total,
        vw_es_bps=vw("es_bps"),
        **{f"markout_{h}": vw(f"markout_{h}_bps") for h in horizons},
    )
    by_bucket = pt.group_by("bucket").agg(**aggs)
    all_row = pt.select(bucket=pl.lit("all"), **aggs)
    order = pl.DataFrame({"bucket": [*BUCKETS, "all"]})
    return order.join(pl.concat([by_bucket, all_row]), on="bucket", how="left")


def markout_cis(pt: pl.DataFrame, horizons: list[str], block: str) -> pl.DataFrame:
    """Bootstrap CIs per bucket and horizon, in the layout of the saved curves.

    Also adds MERGED = "1 through" + "2-10 through", bootstrapped from the trades
    themselves (same blocks and seed, so the same draws), not combined from the two CIs.
    """
    values = {f"markout_{h}_bps": h for h in horizons}
    blocked = pt.with_columns(
        block=pl.col("time").dt.truncate(block),
        merged=pl.col("bucket").replace({"1 through": MERGED, "2-10 through": MERGED}),
    )

    def boot(group_col: str) -> pl.DataFrame:
        return block_bootstrap_wmean(
            blocked, list(values), weight_col="notional", block_col="block",
            group_col=group_col, n_boot=N_BOOT, seed=SEED,
        )

    ci = pl.concat([boot("bucket"), boot("merged").filter(pl.col("group") == MERGED)])
    return ci.select(
        pl.col("group").alias("bucket"),
        pl.col("value").replace_strict(values).alias("horizon"),
        pl.col("value").replace_strict({v: horizon_seconds(h) for v, h in values.items()})
        .alias("horizon_s"),
        pl.col("point").alias("vw_markout_bps"),
        "ci_lo",
        "ci_hi",
        pl.col("n").alias("n_aggtrades"),
        pl.col("weight").alias("notional"),
        pl.lit(block).alias("block"),
    ).sort(pl.col("bucket").cast(pl.Enum([*BUCKETS, MERGED, "all"])), "horizon_s")


def with_ci(point: pl.DataFrame, ci: pl.DataFrame, horizons: list[str]) -> pl.DataFrame:
    """Markout table with 'point [lo, hi]' strings for the given horizons."""
    out = point.select("bucket", "n_aggtrades", "notional_share", "vw_es_bps")
    for h in horizons:
        cell = ci.filter(pl.col("horizon") == h).select(
            "bucket",
            pl.format(
                "{} [{}, {}]",
                *[pl.col(c).round(3).cast(pl.String) for c in ("vw_markout_bps", "ci_lo", "ci_hi")],
            ).alias(f"markout_{h}"),
        )
        out = out.join(cell, on="bucket", how="left", maintain_order="left")
    return out


def delta(base: pl.DataFrame, alt: pl.DataFrame, key: str, cols: list[str]) -> pl.DataFrame:
    """alt - base for each column, matched on key."""
    j = base.select(key, *cols).join(
        alt.select(key, *cols), on=key, how="left", suffix="_alt", maintain_order="left"
    )
    # Cast first: counts are UInt32 (pl.len()), and a negative difference would wrap around.
    f = pl.Float64
    return j.select(
        key, *[(pl.col(f"{c}_alt").cast(f) - pl.col(c).cast(f)).alias(f"d_{c}") for c in cols]
    )


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--date", type=date.fromisoformat, default=date(2024, 3, 1))
    parser.add_argument("--start", type=time.fromisoformat, default=time(0, 0))
    parser.add_argument("--end", type=time.fromisoformat, default=None, help="default: 24:00")
    parser.add_argument("--block", default="5m", help="bootstrap block length, e.g. 1m, 5m")
    args = parser.parse_args()
    start = datetime.combine(args.date, args.start, tzinfo=timezone.utc)
    end = (
        datetime.combine(args.date, args.end, tzinfo=timezone.utc)
        if args.end
        else datetime.combine(args.date + timedelta(days=1), time(0), tzinfo=timezone.utc)
    )

    day = load_perp(args.date, symbol=args.symbol)
    # Materialize quotes once, sorted by (time, update_id), so the repeated as-of joins
    # below do not rescan the file.
    quotes = (
        day.quotes.filter(pl.col("time").is_between(start - PAD, end + MAX_HORIZON + PAD))
        .collect()
        .sort(["time", "update_id"])
        .lazy()
    )
    trades_window = day.trades.filter(pl.col("time").is_between(start, end, closed="left"))
    n_window = trades_window.select(pl.len()).collect().item()
    trades = (
        drop_trades_near_end(trades_window, quotes, HORIZONS[-1])
        .collect()
        .sort("first_trade_id")
    )

    # Taker events; quintiles fixed on the full sample so robustness deltas compare
    # the same size buckets.
    ev = group_taker_events(trades.lazy(), tick=day.tick).collect()
    ev = ev.with_columns(
        first_in_ms=pl.int_range(pl.len()).over("time") == 0,
        quintile=pl.format(
            "Q{}", (pl.col("notional").rank("ordinal") - 1) * 5 // pl.len() + 1
        ),
    )
    # Map each aggTrade to its event (events are contiguous trade-id ranges).
    trades = trades.join_asof(
        ev.select(pl.col("first_trade_id").alias("event_id"), "first_in_ms"),
        left_on="first_trade_id",
        right_on="event_id",
        strategy="backward",
    )
    ev = ev.join(
        trades.group_by("event_id").agg(any_off_grid=pl.col("off_grid").any()),
        left_on="first_trade_id",
        right_on="event_id",
    )
    ev = effective_spread_bps(
        attach_mid(ev.lazy().with_columns(price=pl.col("vwap")), quotes)
    ).collect()

    pt = (
        effective_spread_bps(markouts_bps(attach_touch(trades.lazy(), quotes), quotes, HORIZONS))
        .with_columns(bucket=position_bucket(day.tick), notional=pl.col("price") * pl.col("qty"))
        .collect()
    )

    # Baseline + robustness
    ev_base = event_table(ev)
    ev_a = event_table(ev.filter(pl.col("first_in_ms")))
    ev_b = event_table(ev.filter(~pl.col("any_off_grid")))
    mk_base = markout_table(pt, HORIZONS)
    mk_a = markout_table(pt.filter(pl.col("first_in_ms")), HORIZONS)
    mk_b = markout_table(pt.filter(~pl.col("off_grid")), HORIZONS)
    ci = markout_cis(pt, HORIZONS, args.block)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"markouts_{args.symbol}_{args.date}_{args.block}.parquet"
    ci.write_parquet(out_path)

    ev_cols = ["n_events", "mean_es_bps", "vw_es_bps"]
    mk_cols = ["n_aggtrades", "vw_es_bps", *[f"markout_{h}" for h in TABLE_HORIZONS]]
    print(
        f"\n=== {args.symbol} perp (tick {day.tick:g}) "
        f"{start:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M} UTC ===\n"
        f"aggTrades in window: {n_window} | dropped near end ({HORIZONS[-1]}): "
        f"{n_window - trades.height} | taker events: {ev.height} | "
        f"events without a mid: {ev['mid'].null_count()} | "
        f"not first in ms: {(~ev['first_in_ms']).sum()} | with off-grid trade: "
        f"{ev['any_off_grid'].sum()} | off-grid aggTrades: {trades['off_grid'].sum()}"
    )
    cfg = dict(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250, float_precision=3,
               tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True)
    with pl.Config(**cfg):
        print("\nES per taker event by notional quintile (USDT, bps)")
        print(ev_base)
        print(f"\nMaker markouts per aggTrade by position vs touch (notional-weighted, bps, "
              f"95% CI: {args.block} block bootstrap, {N_BOOT} resamples)")
        print(with_ci(mk_base, ci, TABLE_HORIZONS))
        print("\nRobustness (a) only first event in each ms — change vs baseline")
        print(delta(ev_base, ev_a, "quintile", ev_cols))
        print(delta(mk_base, mk_a, "bucket", mk_cols))
        print("\nRobustness (b) without off-grid trades — change vs baseline")
        print(delta(ev_base, ev_b, "quintile", ev_cols))
        print(delta(mk_base, mk_b, "bucket", mk_cols))
    print(f"\nmarkout curves ({len(HORIZONS)} horizons, with CIs) -> {out_path}")


if __name__ == "__main__":
    main()
