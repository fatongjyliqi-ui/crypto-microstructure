"""Quoted spread of the BTCUSDT perp over one day, from bookTicker.

Usage:
    python -m crypto_microstructure.download --date 2024-03-01
    python examples/plot_perp_spread.py --date 2024-03-01
"""

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

from crypto_microstructure.download import DATA_DIR
from crypto_microstructure.spread import add_mid_and_quoted_spread, sample_on_grid

SYMBOL = "BTCUSDT"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

BLUE, BLUE_LIGHT = "#2a78d6", "#9ec5f4"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, default=date(2024, 3, 1))
    plot_day(parser.parse_args().date)


def plot_day(day: date) -> None:
    path = DATA_DIR / "parquet" / "perp_bookTicker" / f"{SYMBOL}-bookTicker-{day}.parquet"
    book = add_mid_and_quoted_spread(pl.scan_parquet(path))

    # Raw event stats (every update counts once). Tick = most common spread, not the
    # smallest: 2024-03-01 has a few hundred off-grid quotes (e.g. ask 61431.87, tick 0.1).
    stats = book.select(
        n=pl.len(),
        tick=pl.col("spread").round(8).mode().first(),
        crossed=(pl.col("spread") <= 0).sum(),
        event_mean_bps=pl.col("spread_bps").mean(),
    ).collect()
    print(stats)

    # Time-weighted view: book state every second, then per-minute mean and max.
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1) - timedelta(seconds=1)
    per_sec = sample_on_grid(book, start, end, every="1s").drop_nulls("spread_bps")
    tick = stats["tick"].item()
    summary = per_sec.select(
        tw_mean_bps=pl.col("spread_bps").mean(),
        share_one_tick=(pl.col("spread") < 1.5 * tick).mean(),
        p99_bps=pl.col("spread_bps").quantile(0.99),
    ).collect()
    print(summary)

    per_min = (
        per_sec.group_by_dynamic("time", every="1m")
        .agg(mean_bps=pl.col("spread_bps").mean(), max_bps=pl.col("spread_bps").max())
        .collect()
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(per_min["time"], per_min["max_bps"], color=BLUE_LIGHT, lw=1, label="1-min max")
    ax.plot(per_min["time"], per_min["mean_bps"], color=BLUE, lw=2, label="1-min mean")
    ax.set_yscale("log")
    ax.set_ylabel("Quoted spread (bps, log scale)", color=INK_MUTED)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    ax.set_xlabel("Time (UTC)", color=INK_MUTED)
    ax.set_xlim(start, start + timedelta(days=1))
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(frameon=False, loc="upper left")
    s = summary.row(0, named=True)
    ax.set_title(
        f"{SYMBOL} perp quoted spread, {day} — time-weighted mean {s['tw_mean_bps']:.4f} bps, "
        f"at 1 tick ({tick:g} USDT) {s['share_one_tick']:.1%} of the time",
        loc="left",
        color=INK,
        fontsize=11,
    )

    PLOTS_DIR.mkdir(exist_ok=True)
    out = PLOTS_DIR / f"perp_spread_{day}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
