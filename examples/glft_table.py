"""Model vs observed vs break-even spread per perp, all in ticks.

Usage:
    python examples/fill_intensity.py --date 2024-03-01   # writes the intensity inputs
    python examples/glft_table.py --date 2024-03-01

(a) GLFT optimal spread at q = 0 (exact stationary solution, inventory bound Q lots),
    kappa from censoring K = 1, A between A_through and A_touched.
    gamma: realistic values in 1/USDT mapped to model units, plus a stress grid in gamma/kappa.
(b) Observed quoted spread, time-weighted.
(c) Break-even spread for fills at the touch: S_be = 2 E_w[s (m_{t+h} - m_t)] in ticks,
    notional-weighted, with 5-minute block-bootstrap CIs (h = 1 s, 30 s). Before fees.

Units: one lot = LOT_USDT of notional. Wealth in the model is counted in tick x lot, so a
CARA coefficient gamma_usdt (1/USDT) becomes gamma = gamma_usdt * LOT_USDT * tick / price
per tick and lot.
"""

import argparse
from datetime import date, datetime, time, timedelta, timezone

import polars as pl

from crypto_microstructure.download import DATA_DIR
from crypto_microstructure.glft import glft_spread_exact
from crypto_microstructure.load import TICKS, load_perp
from crypto_microstructure.metrics import (
    attach_touch,
    drop_trades_near_end,
    markouts_bps,
    trade_sign,
)
from crypto_microstructure.stats import block_bootstrap_wmean

LOT_USDT = 100.0
Q_LOTS = 1000  # inventory bound: 100k USDT; Q >= 20 changes the spread by < 0.3%
# CARA coefficient in 1/USDT. Assumption: gamma_usdt = RRA / capital with relative risk
# aversion 1-10 and 1e5-1e6 USDT of capital per symbol, i.e. 1e-6 to 1e-4.
GAMMA_USDT = [1e-6, 1e-4]
GAMMA_OVER_KAPPA = [0.01, 0.1]  # stress grid
HORIZONS = ["1s", "30s"]
BLOCK = "5m"
RESULTS_DIR = DATA_DIR / "results"


def observed_spread(quotes: pl.DataFrame, tick: float, end: datetime) -> dict:
    q = quotes.with_columns(
        ticks=((pl.col("ask") - pl.col("bid")) / tick).round(0),
        dur=(pl.col("time").shift(-1).fill_null(end) - pl.col("time")).dt.total_microseconds()
        / 1e6,
    )
    w = q["dur"]
    by_tick = q.group_by("ticks").agg(pl.col("dur").sum()).sort("ticks").with_columns(
        cum=pl.col("dur").cum_sum() / pl.col("dur").sum()
    )
    return dict(
        obs_mean_ticks=float((q["ticks"] * w).sum() / w.sum()),
        obs_median_ticks=float(by_tick.filter(pl.col("cum") >= 0.5)["ticks"][0]),
        obs_share_1tick=float((w * (q["ticks"] == 1)).sum() / w.sum()),
    )


def break_even(trades: pl.LazyFrame, quotes: pl.LazyFrame, tick: float) -> dict:
    """Notional-weighted S_be = 2 s (m_{t+h} - m_t) in ticks over fills at the touch."""
    s = trade_sign(pl.col("is_buyer_maker"))
    touch = pl.when(s == 1).then(pl.col("ask")).otherwise(pl.col("bid"))
    pt = (
        markouts_bps(attach_touch(trades, quotes), quotes, HORIZONS)
        .filter(((s * (pl.col("price") - touch)) / tick).round(0) == 0)
        .with_columns(
            notional=pl.col("price") * pl.col("qty"),
            block=pl.col("time").dt.truncate(BLOCK),
            # pi_h_bps = 2 s (m_{t+h} - m_t) / m_t * 1e4  ->  ticks: * m_t / (tick * 1e4)
            **{
                f"sbe_{h}": pl.col(f"pi_{h}_bps") * pl.col("mid") / (tick * 1e4)
                for h in HORIZONS
            },
        )
        .collect()
    )
    ci = block_bootstrap_wmean(pt, [f"sbe_{h}" for h in HORIZONS], "notional", "block")
    out = {"n_touch_fills": pt.height}
    for r in ci.iter_rows(named=True):
        h = r["value"].removeprefix("sbe_")
        out[f"sbe_{h}"], out[f"sbe_{h}_lo"], out[f"sbe_{h}_hi"] = r["point"], r["ci_lo"], r["ci_hi"]
    return out


def model_spreads(r: dict) -> dict:
    k, sig = r["kappa"], r["sigma_ticks_sqrt_s"]
    tick_rel = r["tick_bps"] * 1e-4
    out = {}
    gammas = [(f"g_usdt_{g:g}", g * LOT_USDT * tick_rel) for g in GAMMA_USDT]
    gammas += [(f"g_over_k_{x:g}", x * k) for x in GAMMA_OVER_KAPPA]
    for label, g in gammas:
        lo = glft_spread_exact(g, k, r["a_touched"], sig, Q_LOTS)  # larger A -> narrower
        hi = glft_spread_exact(g, k, r["a_through"], sig, Q_LOTS)
        out[f"model_{label}"] = f"{lo:.2f}–{hi:.2f}" if abs(hi - lo) >= 0.005 else f"{lo:.2f}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, default=date(2024, 3, 1))
    args = parser.parse_args()
    end = datetime.combine(args.date + timedelta(days=1), time(0), tzinfo=timezone.utc)

    inten = pl.read_parquet(RESULTS_DIR / f"intensity_{args.date}.parquet").filter(
        pl.col("variant") == "all"
    )
    rows = []
    for symbol in TICKS:
        r1 = inten.filter((pl.col("symbol") == symbol) & (pl.col("K") == "1")).row(0, named=True)
        r5 = inten.filter((pl.col("symbol") == symbol) & (pl.col("K") == "5")).row(0, named=True)
        d = load_perp(args.date, symbol=symbol)
        quotes = d.quotes.sort(["time", "update_id"]).collect()
        trades = drop_trades_near_end(d.trades, quotes.lazy(), HORIZONS[-1])
        rows.append(
            dict(
                symbol=symbol,
                tick_bps=r1["tick_bps"],
                kappa_K1=r1["kappa"],
                two_over_kappa_K1=r1["two_over_kappa"],
                two_over_kappa_K1_lo=r1["two_over_kappa_lo"],
                two_over_kappa_K1_hi=r1["two_over_kappa_hi"],
                two_over_kappa_K5=r5["two_over_kappa"],
                **model_spreads(r1),
                **observed_spread(quotes, d.tick, end),
                **break_even(trades, quotes.lazy(), d.tick),
            )
        )
    table = pl.DataFrame(rows).sort("two_over_kappa_K1")
    out = RESULTS_DIR / f"glft_table_{args.date}.parquet"
    table.write_parquet(out)

    def ci(col: str) -> pl.Expr:
        cols = (col, f"{col}_lo", f"{col}_hi")
        return pl.format("{} [{}, {}]", *[pl.col(c).round(2) for c in cols])

    cfg = dict(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=260, float_precision=2,
               tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True)
    with pl.Config(**cfg):
        print(f"Spreads in ticks, {args.date}. Lot {LOT_USDT:g} USDT, Q = {Q_LOTS} lots, "
              f"kappa from K = 1; model range = A_touched–A_through")
        print(table.select(
            "symbol", "tick_bps",
            ci("two_over_kappa_K1").alias("(a) 2/kappa K=1"),
            pl.col("two_over_kappa_K5").alias("2/kappa K=5"),
            *[pl.col(f"model_g_usdt_{g:g}").alias(f"(a) gamma={g:g}/USDT") for g in GAMMA_USDT],
            *[pl.col(f"model_g_over_k_{x:g}").alias(f"(a) gamma/kappa={x:g}")
              for x in GAMMA_OVER_KAPPA],
            pl.col("obs_mean_ticks").alias("(b) mean"),
            pl.col("obs_median_ticks").alias("(b) median"),
            *[ci(f"sbe_{h}").alias(f"(c) S_be {h}") for h in HORIZONS],
            "n_touch_fills",
        ))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
