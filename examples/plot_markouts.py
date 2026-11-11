"""Maker markout curves by position vs the touch, two days side by side.

"1-10 through" is bootstrapped from the trades in perp_taker_events.py (saved as its own
group), not averaged from the "1 through" and "2-10 through" curves.

Usage:
    python examples/perp_taker_events.py --date 2024-03-01   # writes the 5m results
    python examples/perp_taker_events.py --date 2024-03-12
    python examples/plot_markouts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from crypto_microstructure.download import DATA_DIR

DAYS = [("2024-03-01", "2024-03-01 (normal day)"), ("2024-03-12", "2024-03-12 (US CPI 12:30 UTC)")]
BLOCK = "5m"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

# Reference categorical palette, slots 1-3 in fixed order (one per bucket, same on both days).
SERIES = [
    ("at touch", "#2a78d6"),
    ("1-10 through", "#eb6834"),
    (">10 through", "#1baf7a"),
]
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
LABEL_GAP_BPS = 0.19  # minimum vertical spacing of the end-of-line labels


def spread_labels(ys: list[float], gap: float) -> list[float]:
    """Nudge label positions apart (keeping their order) so none are closer than `gap`."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:], strict=False):
        out[b] = max(out[b], out[a] + gap)
    return out


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, facecolor=SURFACE)
    for ax, (day, panel_title) in zip(axes, DAYS, strict=True):
        df = pl.read_parquet(DATA_DIR / "results" / f"markouts_{day}_{BLOCK}.parquet")
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ends = []
        for bucket, color in SERIES:
            s = df.filter(pl.col("bucket") == bucket).sort("horizon_s")
            x = s["horizon_s"].to_numpy()
            ax.fill_between(x, s["ci_lo"], s["ci_hi"], color=color, alpha=0.14, lw=0, zorder=2)
            ax.plot(
                x, s["vw_markout_bps"], color=color, lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=bucket, zorder=3,
            )
            ends.append((bucket, x[-1], s["vw_markout_bps"][-1]))
        # Direct labels at the 60s end, nudged apart where lines end close together.
        ys = spread_labels([y for _, _, y in ends], LABEL_GAP_BPS)
        for (bucket, x_end, _), y in zip(ends, ys, strict=True):
            ax.annotate(bucket, (x_end, y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_2)

        ticks = df["horizon_s"].unique().sort().to_list()
        ax.set_xscale("log")
        ax.set_xticks(ticks, [df.filter(pl.col("horizon_s") == t)["horizon"][0] for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(0.08, 140)  # room on the right for the direct labels
        ax.set_title(panel_title, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_xlabel("Horizon after the trade (log scale)", color=MUTED)
        ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK_2)
    axes[0].set_ylabel("Maker markout (bps of mid, notional-weighted)", color=MUTED)
    # One tick (0.1) at ~61,000 is 0.016 bps of spread, so half of it is ~0.008 bps.
    axes[0].annotate(
        "Half-spread earned at the touch ≈ 0.008 bps\n(invisible at this scale)",
        xy=(0.13, 0.008), xytext=(0.13, 0.9), textcoords="data",
        fontsize=9, color=INK_2, ha="left", va="bottom",
        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, shrinkA=2, shrinkB=0),
    )

    fig.suptitle(
        "Makers lose at every depth on both days; deep fills recover only on CPI day, "
        "and not significantly",
        x=0.06, y=0.985, ha="left", fontsize=14, color=INK,
    )
    fig.text(
        0.06, 0.92,
        "BTCUSDT perp, Binance. Maker P&L per aggTrade vs the mid h later, bucketed by ticks "
        "through the touch. Shaded: 95% CI, 5-min block bootstrap (1000 resamples).\n"
        "On 2024-03-12 the >10-through CI includes 0 from 5 s on.",
        ha="left", va="top", fontsize=9, color=INK_2,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.005), labelcolor=INK_2)
    fig.tight_layout(rect=(0.02, 0.05, 1, 0.91))

    PLOTS_DIR.mkdir(exist_ok=True)
    out = PLOTS_DIR / f"markouts_by_depth_{BLOCK}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
