"""Effective spread and maker markouts of the BTCUSDT perp over one intraday window.

Usage:
    python examples/perp_metrics_window.py --date 2024-03-01 --start 14:00 --end 15:00
"""

import argparse
from datetime import date, datetime, time, timedelta, timezone

import polars as pl

from crypto_microstructure.load import load_perp
from crypto_microstructure.metrics import (
    drop_trades_near_end,
    effective_spread_bps,
    markouts_bps,
)

HORIZONS = ["1s", "5s", "30s"]
MAX_HORIZON = timedelta(seconds=30)
PAD = timedelta(minutes=1)  # quotes before the window, so the first trades have a mid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, default=date(2024, 3, 1))
    parser.add_argument("--start", type=time.fromisoformat, default=time(14, 0))
    parser.add_argument("--end", type=time.fromisoformat, default=time(15, 0))
    args = parser.parse_args()
    start = datetime.combine(args.date, args.start, tzinfo=timezone.utc)
    end = datetime.combine(args.date, args.end, tzinfo=timezone.utc)

    day = load_perp(args.date)
    print(f"off-grid quotes dropped (whole day): {day.n_off_grid_dropped}")

    trades = day.trades.filter(pl.col("time").is_between(start, end, closed="left"))
    quotes = day.quotes.filter(
        pl.col("time").is_between(start - PAD, end + MAX_HORIZON + PAD)
    )
    n_window = trades.select(pl.len()).collect().item()
    trades = drop_trades_near_end(trades, quotes, HORIZONS[-1])

    out = effective_spread_bps(markouts_bps(trades, quotes, HORIZONS)).collect()
    n_near_end = n_window - out.height
    n_null = out["mid"].null_count()
    out = out.drop_nulls("mid")

    notional = pl.col("price") * pl.col("qty")

    def vw(col: str) -> pl.Expr:
        return (pl.col(col) * notional).sum() / notional.sum()

    summary = out.select(
        n_trades=pl.len(),
        n_off_grid=pl.col("off_grid").sum(),
        vw_mean_es_bps=vw("es_bps"),
        share_es_neg=(pl.col("es_bps") < 0).mean(),
    )
    markouts = pl.DataFrame(
        {
            "horizon": HORIZONS,
            "mean_bps": [out[f"markout_{h}_bps"].mean() for h in HORIZONS],
            "vw_mean_bps": [out.select(vw(f"markout_{h}_bps")).item() for h in HORIZONS],
        }
    )

    print(
        f"window {start:%Y-%m-%d %H:%M}–{end:%H:%M} UTC | "
        f"dropped near end: {n_near_end} | without a mid: {n_null}"
    )
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(summary.transpose(include_header=True, column_names=["value"]))
        print(markouts)


if __name__ == "__main__":
    main()
