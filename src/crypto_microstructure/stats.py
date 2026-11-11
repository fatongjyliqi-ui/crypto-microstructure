"""Block bootstrap for weighted means."""

from __future__ import annotations

import numpy as np
import polars as pl


def block_bootstrap_wmean(
    df: pl.DataFrame,
    value_cols: list[str],
    weight_col: str,
    block_col: str,
    group_col: str | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> pl.DataFrame:
    """Percentile CIs of weighted means sum(w * x) / sum(w), resampling whole blocks.

    Per block b we keep W[b] = sum(w) and N[b, v] = sum(w * x_v). A resample draws
    n_blocks blocks with replacement; with C[r, b] the number of times block b was drawn
    in resample r, the statistic is (C @ N)[r, v] / (C @ W)[r]. Observations within a
    block stay together, so dependence inside a block is preserved.

    With `group_col`, each group gets its own rows plus a group "all" over every row;
    all groups and value columns use the same draws. A block where a group has no rows
    contributes 0 to both sums; a resample with zero total weight is ignored (NaN).

    Returns one row per (group, value): group, value, point, ci_lo, ci_hi, n, weight.
    """
    blocks = df.select(pl.col(block_col).unique().sort())
    n_blocks = blocks.height
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    counts = np.stack([np.bincount(d, minlength=n_blocks) for d in draws])  # (n_boot, n_blocks)
    pct = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]

    subsets: list[tuple[str, pl.DataFrame]] = []
    if group_col is not None:
        for g in df[group_col].unique().sort():
            subsets.append((g, df.filter(pl.col(group_col) == g)))
    subsets.append(("all", df))

    w = pl.col(weight_col)
    rows = []
    for g, sub in subsets:
        per_block = blocks.join(
            sub.group_by(block_col).agg(
                _w=w.sum(), _n=pl.len(), **{v: (pl.col(v) * w).sum() for v in value_cols}
            ),
            on=block_col,
            how="left",
            maintain_order="left",
        ).fill_null(0)
        W = per_block["_w"].to_numpy()
        N = per_block.select(value_cols).to_numpy()  # (n_blocks, n_values)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (counts @ N) / (counts @ W)[:, None]  # (n_boot, n_values)
        point = N.sum(axis=0) / W.sum()
        lo, hi = np.nanpercentile(boot, pct, axis=0)
        for i, v in enumerate(value_cols):
            rows.append(
                dict(
                    group=g,
                    value=v,
                    point=float(point[i]),
                    ci_lo=float(lo[i]),
                    ci_hi=float(hi[i]),
                    n=int(per_block["_n"].sum()),
                    weight=float(W.sum()),
                )
            )
    return pl.DataFrame(rows)
