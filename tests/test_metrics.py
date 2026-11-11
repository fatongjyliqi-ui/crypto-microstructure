from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from crypto_microstructure.metrics import (
    attach_mid,
    drop_trades_near_end,
    effective_spread_bps,
    markouts_bps,
    trade_sign,
)

T0 = datetime(2024, 3, 1, tzinfo=timezone.utc)


def ms(x: int) -> datetime:
    return T0 + timedelta(milliseconds=x)


@pytest.fixture
def book():
    quotes = pl.LazyFrame(
        {
            "time": [ms(1000), ms(1500), ms(2500)],
            "bid": [60000.0, 60000.1, 60001.0],
            "ask": [60000.1, 60000.2, 60001.1],
        }
    )
    trades = pl.LazyFrame(
        {
            "time": [ms(1500), ms(2000)],
            "price": [60000.1, 60000.1],
            "qty": [1.0, 1.0],
            "is_buyer_maker": [False, True],
        }
    )
    return trades, quotes


def test_trade_sign():
    df = pl.DataFrame({"m": [False, True]})
    assert df.select(trade_sign(pl.col("m"))).to_series().to_list() == [1, -1]


def test_mid_is_strictly_before_trade(book):
    trades, quotes = book
    mids = attach_mid(trades, quotes).collect()["mid"].to_list()
    # Trade 1 at 1500 must NOT see the quote stamped 1500
    assert mids == pytest.approx([60000.05, 60000.15])


def test_effective_spread(book):
    trades, quotes = book
    es = effective_spread_bps(attach_mid(trades, quotes)).collect()["es_bps"]
    expected = [2 * 0.05 / 60000.05 * 1e4, 2 * 0.05 / 60000.15 * 1e4]
    assert es.to_list() == pytest.approx(expected, rel=1e-6)
    assert (es > 0).all()  # a taker never trades better than mid at the touch


def test_markouts_1s(book):
    trades, quotes = book
    out = markouts_bps(trades, quotes, ["1s"]).collect()
    expected = [-0.05 / 60000.05 * 1e4, 0.95 / 60000.15 * 1e4]
    assert out["markout_1s_bps"].to_list() == pytest.approx(expected, rel=1e-6)


def test_decomposition_es_equals_rs_plus_pi(book):
    trades, quotes = book
    out = effective_spread_bps(markouts_bps(trades, quotes, ["1s", "5s"])).collect()
    for h in ["1s", "5s"]:
        lhs = out["es_bps"]
        rhs = out[f"rs_{h}_bps"] + out[f"pi_{h}_bps"]
        assert lhs.to_list() == pytest.approx(rhs.to_list(), rel=1e-9)


def test_drop_trades_near_end(book):
    trades, quotes = book
    # Last quote at 2500.
    #   "1s":    1500 + 1000 = 2500 <= 2500 -> keep;  2000 + 1000 = 3000 > 2500 -> drop
    #   "500ms": 2000, 2500 both <= 2500     -> keep both (equality is kept)
    #   "2s":    3500, 4000                  -> drop both
    def kept(h):
        return drop_trades_near_end(trades, quotes, h).collect()["time"].to_list()

    assert kept("1s") == [ms(1500)]
    assert kept("500ms") == [ms(1500), ms(2000)]
    assert kept("2s") == []
