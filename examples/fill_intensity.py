"""Fill intensity lambda(delta) = A exp(-kappa delta) per perp, for the AS/GLFT comparison.

Usage:
    python examples/fill_intensity.py --date 2024-03-01
    python examples/fill_intensity.py --date 2024-03-12 --symbols BTCUSDT XRPUSDT

For every symbol in TICKS: taker events, X = ticks the worst fill lies beyond the touch,
kappa per tick for censoring K = 1, 5, inf with 5-minute block-bootstrap CIs, for all
events and for only the first event in each millisecond (robustness: later events in the
same ms are measured against a touch that may already be stale). Also sigma of the mid
in ticks per sqrt(second) from 1-minute log returns.

Writes data/results/intensity_<date>.parquet and, for BTCUSDT and XRPUSDT,
data/results/depth_survival_<symbol>_<date>.parquet (empirical P(X >= k) and theta^k).
"""

import argparse
import math
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import polars as pl

from crypto_microstructure.download import DATA_DIR
from crypto_microstructure.intensity import fit_intensity, geometric_theta, ticks_beyond_touch
from crypto_microstructure.load import TICKS, load_perp
from crypto_microstructure.metrics import attach_touch, group_taker_events

K_VALUES = [1, 5, None]  # None = no censoring
BLOCK = "5m"
SURVIVAL_SYMBOLS = ["BTCUSDT", "XRPUSDT"]
K_PLOT_MAX = 100
RESULTS_DIR = DATA_DIR / "results"


def k_label(k: int | None) -> str:
    return "inf" if k is None else str(k)


def one_tick_exposure_s(quotes: pl.DataFrame, tick: float, end: datetime) -> float:
    """Seconds the book had a one-tick spread (each quote live until the next one)."""
    return (
        quotes.with_columns(
            dur=(pl.col("time").shift(-1).fill_null(end) - pl.col("time"))
            .dt.total_microseconds()
            / 1e6,
            st=((pl.col("ask") - pl.col("bid")) / tick).round(0),
        )
        .filter(pl.col("st") == 1)["dur"]
        .sum()
    )


def sigma_ticks_per_sqrt_s(quotes: pl.DataFrame, tick: float, day: date, end: datetime) -> float:
    """Std of 1-minute mid changes in ticks, divided by sqrt(60) (arithmetic BM in AS/GLFT).

    Uses log returns times the mean mid, which equals price changes to first order.
    """
    grid = pl.DataFrame(
        {
            "time": pl.datetime_range(
                datetime.combine(day, time(0, 1), tzinfo=timezone.utc), end, "1m",
                eager=True, time_zone="UTC",
            )
        }
    ).with_columns(pl.col("time").dt.cast_time_unit("us"))
    mid = grid.join_asof(
        quotes.select("time", ((pl.col("bid") + pl.col("ask")) / 2).alias("mid")),
        on="time", strategy="backward",
    )["mid"].to_numpy()
    sd_price = np.std(np.diff(np.log(mid)), ddof=1) * mid.mean()
    return float(sd_price / tick / math.sqrt(60))


def survival(x: np.ndarray, variant: str) -> pl.DataFrame:
    ks = np.arange(0, min(int(x.max()), K_PLOT_MAX) + 1)
    out = {
        "variant": [variant] * len(ks),
        "k": ks,
        "n_ge_k": [(x >= k).sum() for k in ks],
        "p_emp": [(x >= k).mean() for k in ks],
    }
    for kk in K_VALUES:
        out[f"p_geom_K{k_label(kk)}"] = geometric_theta(x, kk) ** ks
    return pl.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, default=date(2024, 3, 1))
    # The output file holds only the symbols of this run.
    parser.add_argument("--symbols", nargs="+", default=list(TICKS), choices=list(TICKS))
    args = parser.parse_args()
    end = datetime.combine(args.date + timedelta(days=1), time(0), tzinfo=timezone.utc)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for symbol in args.symbols:
        d = load_perp(args.date, symbol=symbol)
        quotes = d.quotes.sort(["time", "update_id"]).collect()
        exposure = one_tick_exposure_s(quotes, d.tick, end)
        sigma = sigma_ticks_per_sqrt_s(quotes, d.tick, args.date, end)
        tick_bps = float((d.tick / ((quotes["bid"] + quotes["ask"]) / 2)).mean() * 1e4)

        events = group_taker_events(d.trades, tick=d.tick).with_columns(
            # events come in trade-id order, so rank 0 within a timestamp is the first one
            first_in_ms=pl.int_range(pl.len()).over("time") == 0
        )
        events = ticks_beyond_touch(attach_touch(events, quotes.lazy()), d.tick).collect()
        used = (pl.col("spread_ticks") == 1) & (pl.col("x") >= 0)

        surv = []
        for variant, ev in [("all", events), ("first_in_ms", events.filter("first_in_ms"))]:
            for k in K_VALUES:
                f = fit_intensity(ev, exposure, k_max=k, block=BLOCK).filter(
                    pl.col("side") == "both"
                ).row(0, named=True)
                rows.append(
                    dict(
                        symbol=symbol, tick=d.tick, tick_bps=tick_bps,
                        sigma_ticks_sqrt_s=sigma, exposure_h=exposure / 3600,
                        variant=variant, K=k_label(k),
                        n_events=f["n_events"], n_beyond_touch=f["n_beyond_touch"],
                        lambda_per_s=f["lambda_per_s"], theta=f["theta"],
                        kappa=f["kappa"], kappa_lo=f["kappa_lo"], kappa_hi=f["kappa_hi"],
                        two_over_kappa=f["two_over_kappa"],
                        two_over_kappa_lo=2 / f["kappa_hi"], two_over_kappa_hi=2 / f["kappa_lo"],
                        a_touched=f["a_touched"], a_through=f["a_through"],
                    )
                )
            if symbol in SURVIVAL_SYMBOLS:
                surv.append(survival(ev.filter(used)["x"].to_numpy(), variant))
        if surv:
            path = RESULTS_DIR / f"depth_survival_{symbol}_{args.date}.parquet"
            pl.concat(surv).write_parquet(path)
            print(f"wrote {path}")

    table = pl.DataFrame(rows)
    out = RESULTS_DIR / f"intensity_{args.date}.parquet"
    table.write_parquet(out)

    def ci(col: str) -> pl.Expr:
        return pl.format(
            "{} [{}, {}]",
            *[pl.col(c).round(2) for c in (col, f"{col}_lo", f"{col}_hi")],
        ).alias(col)

    cfg = dict(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250, float_precision=3,
               tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True)
    with pl.Config(**cfg):
        print(f"\nkappa per tick by censoring K (both sides pooled), {args.date}, "
              f"95% CI {BLOCK} block bootstrap")
        print(
            table.filter(pl.col("variant") == "all")
            .pivot("K", index="symbol", values="kappa")
            .join(
                table.filter(pl.col("variant") == "all").select(
                    "symbol", "K", ci("kappa")
                ).pivot("K", index="symbol", values="kappa"),
                on="symbol", suffix="_ci",
            )
            .sort("1", descending=True)
            .select("symbol", "1_ci", "5_ci", "inf_ci")
            .rename({"1_ci": "K=1", "5_ci": "K=5", "inf_ci": "K=inf"})
        )
        print("\n2/kappa in ticks (risk-neutral model spread), all events vs first event per ms")
        print(
            table.select("symbol", "variant", "K", ci("two_over_kappa"))
            .pivot(["variant", "K"], index="symbol", values="two_over_kappa")
        )
        print("\nInputs for GLFT (K=1, all events)")
        print(
            table.filter((pl.col("variant") == "all") & (pl.col("K") == "1")).select(
                "symbol", "tick_bps", "sigma_ticks_sqrt_s", "lambda_per_s", "kappa",
                "a_touched", "a_through", "n_events", "n_beyond_touch", "exposure_h",
            )
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
