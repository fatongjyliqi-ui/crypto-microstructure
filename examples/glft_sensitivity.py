"""Sensitivity of the exact GLFT spread at q = 0 to the inventory bound Q and the lot size L.

Usage:
    python examples/glft_sensitivity.py --date 2024-03-01
    python examples/glft_sensitivity.py --date 2024-03-12

Reads data/results/intensity_<date>.parquet (kappa from K = 1, A = A_touched) and prints
two tables in ticks:
  1. Q in {1, 5, 10, 20, 1000} at lot 100 USDT, for the realistic gammas and gamma/kappa = 0.1.
     Check: as gamma -> 0 the spread tends to (2/kappa) (1 + log(1 / cos(pi / (2Q + 2)))).
  2. Lot size L in {50, 100, 1000, 10000} USDT with the inventory bound fixed at
     Q L = 100k USDT, gamma_usdt = 1e-4 (the larger realistic value).
"""

import argparse
import math
from datetime import date

import polars as pl

from crypto_microstructure.download import DATA_DIR
from crypto_microstructure.glft import glft_spread_exact

SYMBOLS = ["BTCUSDT", "XRPUSDT"]
Q_VALUES = [1, 5, 10, 20, 1000]
LOT_USDT = 100.0
GAMMA_USDT = [1e-6, 1e-4]
# Smaller lots mean Q > 2000 and a dense eigendecomposition of size 2Q + 1 (slow).
LOTS_USDT = [50.0, 100.0, 1000.0, 10000.0]
INVENTORY_USDT = 1e5
RESULTS_DIR = DATA_DIR / "results"


def cos_limit(kappa: float, Q: int) -> float:
    return 2 / kappa * (1 + math.log(1 / math.cos(math.pi / (2 * Q + 2))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, default=date(2024, 3, 1))
    args = parser.parse_args()
    inten = pl.read_parquet(RESULTS_DIR / f"intensity_{args.date}.parquet").filter(
        (pl.col("variant") == "all") & (pl.col("K") == "1")
    )

    q_rows, l_rows = [], []
    for symbol in SYMBOLS:
        r = inten.filter(pl.col("symbol") == symbol).row(0, named=True)
        k, sig, A = r["kappa"], r["sigma_ticks_sqrt_s"], r["a_touched"]
        tick_rel = r["tick_bps"] * 1e-4
        gammas = [(f"gamma={g:g}/USDT", g * LOT_USDT * tick_rel) for g in GAMMA_USDT]
        gammas.append(("gamma/kappa=0.1", 0.1 * k))
        q_rows.append(dict(symbol=symbol, gamma="limit gamma->0",
                           **{f"Q={Q}": cos_limit(k, Q) for Q in Q_VALUES}))
        for label, g in gammas:
            q_rows.append(dict(symbol=symbol, gamma=label,
                               **{f"Q={Q}": glft_spread_exact(g, k, A, sig, Q) for Q in Q_VALUES}))
        for lot in LOTS_USDT:
            Q = int(INVENTORY_USDT / lot)
            g = 1e-4 * lot * tick_rel
            l_rows.append(dict(symbol=symbol, lot_usdt=lot, Q=Q, gamma_model=g,
                               spread=glft_spread_exact(g, k, A, sig, Q), two_over_kappa=2 / k))

    cfg = dict(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=200, float_precision=4,
               tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True)
    with pl.Config(**cfg):
        print(f"Exact GLFT spread at q = 0 in ticks by inventory bound Q, lot {LOT_USDT:g} USDT, "
              f"{args.date}")
        print(pl.DataFrame(q_rows))
        print(f"\nBy lot size L, Q = {INVENTORY_USDT:g} USDT / L, gamma_usdt = 1e-4")
        print(pl.DataFrame(l_rows).with_columns(pl.col("gamma_model").map_elements(
            lambda x: f"{x:.2e}", return_dtype=pl.String)))


if __name__ == "__main__":
    main()
