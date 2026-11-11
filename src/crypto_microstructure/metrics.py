"""Trade-level microstructure metrics: effective spread, markouts, decomposition."""

import polars as pl

BPS = 1e4


def trade_sign(is_buyer_maker: pl.Expr) -> pl.Expr:
    """Aggressor sign: +1 buyer-initiated, -1 seller-initiated.

    is_buyer_maker=True means the buyer was resting, so the seller was the taker.
    """
    return pl.when(is_buyer_maker).then(-1).otherwise(1).cast(pl.Int8)


def attach_mid(
    trades: pl.LazyFrame,
    quotes: pl.LazyFrame,
    offset: str = "0s",
    name: str = "mid",
) -> pl.LazyFrame:
    """Attach the mid of the last quote strictly before trade time + offset.

    Strictness is implemented by shifting the lookup key back by 1 microsecond,
    so a quote stamped exactly at the lookup time is excluded.
    """
    q = quotes.select(
        pl.col("time").alias("_qtime"),
        ((pl.col("bid") + pl.col("ask")) / 2).alias(name),
    ).sort("_qtime")
    t = trades.with_columns(
        pl.col("time").dt.offset_by(offset).dt.offset_by("-1us").alias("_key")
    ).sort("_key")
    return t.join_asof(
        q, left_on="_key", right_on="_qtime", strategy="backward"
    ).drop("_key", "_qtime")


def drop_trades_near_end(
    trades: pl.LazyFrame, quotes: pl.LazyFrame, max_horizon: str
) -> pl.LazyFrame:
    """Drop trades whose t + max_horizon is after the last quote.

    For those trades the backward as-of join would return the last quote of the file
    as m_{t+h}, i.e. a stale mid rather than the mid at t + h.
    """
    last = quotes.select(pl.col("time").max().alias("_last_quote"))
    return (
        trades.join(last, how="cross")
        .filter(pl.col("time").dt.offset_by(max_horizon) <= pl.col("_last_quote"))
        .drop("_last_quote")
    )


def effective_spread_bps(trades_with_mid: pl.LazyFrame) -> pl.LazyFrame:
    """ES = 2 s (p - m_t) / m_t, in bps."""
    s = trade_sign(pl.col("is_buyer_maker"))
    return trades_with_mid.with_columns(
        (2 * s * (pl.col("price") - pl.col("mid")) / pl.col("mid") * BPS).alias("es_bps")
    )


def markouts_bps(
    trades: pl.LazyFrame, quotes: pl.LazyFrame, horizons: list[str]
) -> pl.LazyFrame:
    """Maker markout, realized spread and price impact per horizon, in bps of m_t.

    markout = s (p - m_{t+h}) / m_t        (maker's P&L per unit)
    RS      = 2 s (p - m_{t+h}) / m_t
    PI      = 2 s (m_{t+h} - m_t) / m_t    so that ES = RS + PI
    """
    out = attach_mid(trades, quotes)
    s = trade_sign(pl.col("is_buyer_maker"))
    for h in horizons:
        mh = f"mid_{h}"
        out = attach_mid(out, quotes, offset=h, name=mh)
        out = out.with_columns(
            (s * (pl.col("price") - pl.col(mh)) / pl.col("mid") * BPS).alias(f"markout_{h}_bps"),
            (2 * s * (pl.col("price") - pl.col(mh)) / pl.col("mid") * BPS).alias(f"rs_{h}_bps"),
            (2 * s * (pl.col(mh) - pl.col("mid")) / pl.col("mid") * BPS).alias(f"pi_{h}_bps"),
        )
    return out
