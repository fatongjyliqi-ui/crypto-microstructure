"""Time-weighted quoted spread in ticks, tick size in bps and 1-minute volatility per perp.

Usage:
    python examples/spread_in_ticks.py --date 2024-03-01

Each quote is weighted by how long it was live (until the next quote; the last one until
the end of the day). Spread in ticks = round((ask - bid) / tick).
"""

import argparse
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import polars as pl

from crypto_microstructure.load import TICKS, load_perp

BPS = 1e4


def weighted_quantile(values: pl.Series, weights: pl.Series, q: float) -> float:
    """Smallest value whose cumulative weight share is >= q (values are discrete ticks)."""
    df = (
        pl.DataFrame({"v": values, "w": weights})
        .group_by("v")
        .agg(pl.col("w").sum())
        .sort("v")
        .with_columns(cum=pl.col("w").cum_sum() / pl.col("w").sum())
    )
    return df.filter(pl.col("cum") >= q)["v"][0]


def spread_profile(symbol: str, day: date) -> dict:
    d = load_perp(day, symbol=symbol)
    end = datetime.combine(day + timedelta(days=1), time(0), tzinfo=timezone.utc)
    q = (
        d.quotes.sort(["time", "update_id"])
        .with_columns(
            mid=(pl.col("bid") + pl.col("ask")) / 2,
            ticks=((pl.col("ask") - pl.col("bid")) / d.tick).round(0).cast(pl.Int64),
            dur=(pl.col("time").shift(-1).fill_null(end) - pl.col("time"))
            .dt.total_microseconds()
            / 1e6,
        )
        .select("time", "mid", "ticks", "dur")
        .collect()
    )
    w = pl.col("dur")

    def share(cond: pl.Expr) -> pl.Expr:
        return (w * cond.cast(pl.Float64)).sum() / w.sum()

    t = pl.col("ticks")
    stats = q.select(
        n_quotes=pl.len(),
        tick_bps=(w * d.tick / pl.col("mid")).sum() / w.sum() * BPS,
        share_le0=share(t <= 0),
        share_1=share(t == 1),
        share_2=share(t == 2),
        share_3_5=share(t.is_between(3, 5)),
        share_gt5=share(t > 5),
        mean_ticks=(w * t).sum() / w.sum(),
    ).row(0, named=True)
    stats["median_ticks"] = weighted_quantile(q["ticks"], q["dur"], 0.5)
    stats["p90_ticks"] = weighted_quantile(q["ticks"], q["dur"], 0.9)

    # 1-minute log returns of the mid: last mid at or before each minute mark.
    start = datetime.combine(day, time(0, 1), tzinfo=timezone.utc)
    grid = pl.DataFrame(
        {"time": pl.datetime_range(start, end, "1m", eager=True, time_zone="UTC")}
    ).with_columns(pl.col("time").dt.cast_time_unit("us"))
    mids = grid.join_asof(q.select("time", "mid"), on="time", strategy="backward")["mid"]
    r = np.diff(np.log(mids.to_numpy()))
    stats["vol_1m_bps"] = float(np.std(r, ddof=1) * BPS)
    stats["vol_1m_ticks"] = stats["vol_1m_bps"] / stats["tick_bps"]
    stats["n_returns"] = len(r)
    return {"symbol": symbol, "tick": d.tick, **stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, default=date(2024, 3, 1))
    args = parser.parse_args()

    rows = [spread_profile(s, args.date) for s in TICKS]
    table = pl.DataFrame(rows).sort(["median_ticks", "mean_ticks"], descending=True)
    cfg = dict(tbl_cols=-1, tbl_width_chars=250, float_precision=3,
               tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True)
    with pl.Config(**cfg):
        print(f"Quoted spread in ticks, time-weighted, {args.date} (USD-M perps)")
        print(table.select(
            "symbol", pl.col("tick").map_elements(lambda x: f"{x:g}", return_dtype=pl.String),
            "tick_bps", "median_ticks", "p90_ticks", "mean_ticks",
            "share_1", "share_2", "share_3_5", "share_gt5", "share_le0",
            "vol_1m_bps", "vol_1m_ticks", "n_quotes",
        ))


if __name__ == "__main__":
    main()
