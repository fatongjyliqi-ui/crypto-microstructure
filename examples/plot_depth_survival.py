"""Empirical P(X >= k) of taker-event depth vs geometric fits, BTCUSDT and XRPUSDT.

X = ticks the worst fill of a taker event lies beyond the touch (one-tick spread before the
event). Lines: theta_K^k for the censored MLE with K = 1, 5, inf.

Usage:
    python examples/fill_intensity.py --date 2024-03-01   # writes depth_survival_*.parquet
    python examples/plot_depth_survival.py --date 2024-03-01
"""

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import MaxNLocator

from crypto_microstructure.download import DATA_DIR

SYMBOLS = ["BTCUSDT", "XRPUSDT"]
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

# Same reference palette as plot_markouts.py: categorical slots 1-3 in fixed order.
FITS = [
    ("p_geom_K1", "geometric, K = 1", "#2a78d6"),
    ("p_geom_K5", "geometric, K = 5", "#eb6834"),
    ("p_geom_Kinf", "geometric, K = ∞", "#1baf7a"),
]
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
Y_MIN = 3e-7  # fitted curves below this are clipped by the axis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, default=date(2024, 3, 1))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, facecolor=SURFACE)
    for ax, symbol in zip(axes, SYMBOLS, strict=True):
        df = (
            pl.read_parquet(DATA_DIR / "results" / f"depth_survival_{symbol}_{args.date}.parquet")
            .filter(pl.col("variant") == "all")
            .sort("k")
        )
        k = df["k"].to_numpy()
        ax.set_facecolor(SURFACE)
        for col, label, color in FITS:
            ax.plot(k, df[col].to_numpy(), color=color, lw=2, label=label, zorder=2)
        ax.plot(
            k, df["p_emp"].to_numpy(), ls="none", marker="o", ms=5, color=INK,
            markeredgecolor=SURFACE, markeredgewidth=1.0, label="empirical", zorder=3,
        )
        n = df["n_ge_k"][0]
        ax.set_title(f"{symbol}  (n = {n:,} events)", loc="left", fontsize=11, color=INK, pad=8)
        ax.set_yscale("log")
        ax.set_ylim(Y_MIN, 1.6)
        ax.set_xlim(-0.5, k.max() + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("k = ticks beyond the touch", color=MUTED)
        ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK_2)
    axes[0].set_ylabel("P(X ≥ k)  (log scale)", color=MUTED)

    fig.suptitle(
        "Depth beyond the touch has a heavy tail: no single geometric rate fits both the "
        "first step and the tail",
        x=0.06, y=0.985, ha="left", fontsize=14, color=INK,
    )
    fig.text(
        0.06, 0.92,
        f"Binance USD-M perps, {args.date}. X = ticks the worst fill of a taker event lies "
        "beyond the touch, events after a one-tick spread. Lines: θ^k from the MLE censored "
        "at K.\nK = 1 matches P(X ≥ 1) exactly and is the main specification; larger K "
        "trades the first step for the tail. BTCUSDT is cut at k = 100.",
        ha="left", va="top", fontsize=9, color=INK_2,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    order = [3, 0, 1, 2]  # empirical first
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005), labelcolor=INK_2)
    fig.tight_layout(rect=(0.02, 0.05, 1, 0.91))

    PLOTS_DIR.mkdir(exist_ok=True)
    out = PLOTS_DIR / f"depth_survival_{args.date}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
