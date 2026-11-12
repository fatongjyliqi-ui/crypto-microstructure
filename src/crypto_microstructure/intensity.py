"""Fill intensity lambda(delta) = A exp(-kappa delta) estimated from taker events.

Setting (Avellaneda-Stoikov; Gueant-Lehalle-Fernandez-Tapia): a limit order at distance
delta from the current mid is filled at rate lambda(delta). With a one-tick spread the mid
sits half a tick inside each touch, so the quotable distances are delta_k = k + 1/2 ticks,
k = 0, 1, 2, ... = ticks beyond the touch.

Per side, let Lambda be the rate of taker events (per second of exposure) and X the number
of ticks the event's worst fill lies beyond the touch just before the event. Two readings
of "an order at level k is filled":
    touched:  lambda(delta_k) = Lambda P(X >= k)      some volume trades at level k
    through:  lambda(delta_k) = Lambda P(X >= k + 1)  level k is fully consumed
"touched" overstates the fill rate of a new order at the back of the queue, "through"
understates it; the truth for a given queue position lies in between.

If X is geometric, P(X >= k) = theta^k with theta = exp(-kappa) (kappa per tick). Then both
readings are exponential in delta with the same kappa, and since delta_k = k + 1/2,
    A_touched = Lambda exp(kappa / 2),   A_through = Lambda exp(-kappa / 2).

MLE of theta, censored at K (only the shape on k < K is used; X >= K counts as ">= K"):
    log L = S log(theta) + N_<K log(1 - theta),  S = sum_i min(X_i, K),  N_<K = #{X_i < K}
    => theta_hat = S / (S + N_<K).
Without censoring (K = inf): theta_hat = Xbar / (1 + Xbar).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from crypto_microstructure.metrics import trade_sign
from crypto_microstructure.stats import block_bootstrap_wmean


def ticks_beyond_touch(events: pl.LazyFrame, tick: float) -> pl.LazyFrame:
    """Add spread_ticks (just before the event) and x = ticks the worst fill lies beyond
    the touch (0 = at the touch, x < 0 = inside the touch, i.e. the quote was stale).

    Expects worst_price, is_buyer_maker (from group_taker_events) and bid, ask (from
    attach_touch).
    """
    s = trade_sign(pl.col("is_buyer_maker"))
    touch = pl.when(s == 1).then(pl.col("ask")).otherwise(pl.col("bid"))
    return events.with_columns(
        spread_ticks=((pl.col("ask") - pl.col("bid")) / tick).round(0).cast(pl.Int64),
        x=(s * (pl.col("worst_price") - touch) / tick).round(0).cast(pl.Int64),
    )


def geometric_theta(x: np.ndarray, k_max: int | None = None) -> float:
    """Censored MLE theta_hat = S / (S + N_<K) of P(X >= k) = theta^k (see module doc)."""
    x = np.asarray(x)
    if (x < 0).any():
        raise ValueError("x must be >= 0")
    if k_max is None:
        s, n_below = x.sum(), len(x)
    else:
        s, n_below = np.minimum(x, k_max).sum(), (x < k_max).sum()
    return float(s / (s + n_below))


def _kappa(theta: float) -> float:
    return math.inf if theta == 0 else -math.log(theta)


def fit_intensity(
    events: pl.DataFrame,
    exposure_s: float,
    k_max: int | None = None,
    block: str = "5m",
    n_boot: int = 1000,
    seed: int = 0,
) -> pl.DataFrame:
    """Fit lambda(delta) = A exp(-kappa delta) per side; delta and kappa in ticks.

    events: output of ticks_beyond_touch (needs time, is_buyer_maker, spread_ticks, x).
    Only events with a one-tick spread before them and x >= 0 are used, so delta_k = k + 1/2.
    exposure_s: seconds the book had a one-tick spread (the time those events come from).

    Rows: side "buy" (buy takers, i.e. fills of ask quotes), "sell" (bid quotes) and
    "both" (pooled; Lambda is then the average per side, n / (2 exposure_s)).
    kappa CI: block bootstrap of theta = sum(a) / sum(b), a = min(x, K), b = a + 1{x < K},
    written as a weighted mean of a/b with weight b; kappa = -log(theta) is decreasing, so
    the theta percentiles map to kappa percentiles in reverse order.
    """
    used = events.filter((pl.col("spread_ticks") == 1) & (pl.col("x") >= 0))
    k = math.inf if k_max is None else k_max
    a = pl.col("x").clip(upper_bound=k) if k_max is not None else pl.col("x")
    b = a + (pl.col("x") < k).cast(pl.Int64)
    boot_df = used.with_columns(
        side=pl.when(pl.col("is_buyer_maker")).then(pl.lit("sell")).otherwise(pl.lit("buy")),
        _blk=pl.col("time").dt.truncate(block),
        _b=b.cast(pl.Float64),
        _r=(a / b).cast(pl.Float64),
    )
    ci = block_bootstrap_wmean(
        boot_df, ["_r"], "_b", "_blk", group_col="side", n_boot=n_boot, seed=seed
    )
    ci = {r["group"]: r for r in ci.iter_rows(named=True)}

    rows = []
    for side in ["buy", "sell", "both"]:
        sub = boot_df if side == "both" else boot_df.filter(pl.col("side") == side)
        x = sub["x"].to_numpy()
        lam = len(x) / exposure_s / (2 if side == "both" else 1)
        theta = geometric_theta(x, k_max)
        kappa = _kappa(theta)
        c = ci["all" if side == "both" else side]
        rows.append(
            dict(
                side=side,
                n_events=len(x),
                n_beyond_touch=int((x >= 1).sum()),
                lambda_per_s=lam,
                theta=theta,
                kappa=kappa,
                kappa_lo=_kappa(c["ci_hi"]),
                kappa_hi=_kappa(c["ci_lo"]),
                a_touched=lam * math.exp(kappa / 2) if math.isfinite(kappa) else math.inf,
                a_through=lam * math.exp(-kappa / 2),
                two_over_kappa=2 / kappa,
            )
        )
    return pl.DataFrame(rows)
