"""Trade-level microstructure metrics: effective spread, markouts, decomposition."""

import polars as pl

from crypto_microstructure.load import TICK

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

    Several quotes can share a timestamp; the as-of join takes the last of them in
    sort order. If `quotes` has an `update_id`, ties are ordered by it (the exchange's
    sequence number); otherwise their input order is kept.
    """
    has_update_id = "update_id" in quotes.collect_schema().names()
    q = quotes.select(
        pl.col("time").alias("_qtime"),
        ((pl.col("bid") + pl.col("ask")) / 2).alias(name),
        *(["update_id"] if has_update_id else []),
    )
    if has_update_id:
        q = q.sort(["_qtime", "update_id"]).drop("update_id")
    else:
        q = q.sort("_qtime", maintain_order=True)
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


def group_taker_events(trades: pl.LazyFrame, tick: float = TICK) -> pl.LazyFrame:
    """Group aggTrades into taker events (one aggressive order sweeping the book).

    Rows sorted by trade id. A new event starts when any of these changes vs the row above:
      - time               (a taker order fills within one engine timestamp)
      - side               (is_buyer_maker)
      - trade-id continuity (first_trade_id != previous last_trade_id + 1)
      - price direction    (taker-side price not strictly worse than the row above)
    The last rule is what separates two orders in the same millisecond: an aggTrade
    already aggregates fills of one taker order at one price, so within an order each
    next aggTrade is at a strictly higher (buy) or lower (sell) price. Trade ids are
    contiguous across back-to-back orders, so continuity alone does not split them.
    Two orders where the second starts beyond the first's last level stay merged:
    public data cannot tell them apart.

    Output per event: time, is_buyer_maker (side), qty, notional, vwap, n_levels,
    ticks_walked, first_trade_id, last_trade_id.
    """
    s = trade_sign(pl.col("is_buyer_maker"))
    new_event = (
        (pl.col("time") != pl.col("time").shift())
        | (pl.col("is_buyer_maker") != pl.col("is_buyer_maker").shift())
        | (pl.col("first_trade_id") != pl.col("last_trade_id").shift() + 1)
        | (s * (pl.col("price") - pl.col("price").shift()) <= 0)
    ).fill_null(True)
    return (
        trades.sort("first_trade_id")
        .with_columns(_event=new_event.cum_sum())
        .group_by("_event", maintain_order=True)
        .agg(
            pl.col("time").first(),
            pl.col("is_buyer_maker").first(),
            pl.col("qty").sum(),
            (pl.col("price") * pl.col("qty")).sum().alias("notional"),
            pl.len().alias("n_levels"),
            ((pl.col("price").max() - pl.col("price").min()) / tick)
            .round(0)
            .cast(pl.Int64)
            .alias("ticks_walked"),
            pl.col("first_trade_id").first(),
            pl.col("last_trade_id").last(),
        )
        .with_columns(vwap=pl.col("notional") / pl.col("qty"))
        .drop("_event")
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
