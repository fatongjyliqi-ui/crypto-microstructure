from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from crypto_microstructure.metrics import (
    attach_mid,
    drop_trades_near_end,
    effective_spread_bps,
    group_taker_events,
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


@pytest.fixture
def sweep() -> pl.LazyFrame:
    # aggTrades in trade-id order. Hand-computed events:
    #   E1  t=1000 buy, ids 1-6, three levels 100.0 / 100.1 / 100.3 (strictly rising)
    #       qty 1 + 2 + 1 = 4, notional 100.0 + 200.2 + 100.3 = 400.5,
    #       vwap 400.5 / 4 = 100.125, ticks walked (100.3 - 100.0) / 0.1 = 3
    #   E2  t=1000 buy, id 7 @ 100.3: same ms, same side, contiguous id, but price not
    #       strictly higher -> a second order that starts at E1's last level
    #   E3  t=1000 sell, id 8 @ 100.2: side changes
    #   E4  t=1000 sell, id 10 @ 100.1: falling price, but id gap (9 missing)
    #   E5  t=2000 sell, id 11 @ 100.0: contiguous and falling, but time changes
    return pl.LazyFrame(
        {
            "time": [ms(1000)] * 6 + [ms(2000)],
            "price": [100.0, 100.1, 100.3, 100.3, 100.2, 100.1, 100.0],
            "qty": [1.0, 2.0, 1.0, 0.5, 1.0, 1.0, 1.0],
            "is_buyer_maker": [False, False, False, False, True, True, True],
            "first_trade_id": [1, 3, 4, 7, 8, 10, 11],
            "last_trade_id": [2, 3, 6, 7, 8, 10, 11],
        }
    )


def test_group_taker_events(sweep):
    ev = group_taker_events(sweep, tick=0.1).collect()
    assert ev.height == 5
    assert ev["first_trade_id"].to_list() == [1, 7, 8, 10, 11]
    assert ev["last_trade_id"].to_list() == [6, 7, 8, 10, 11]
    assert ev["n_levels"].to_list() == [3, 1, 1, 1, 1]
    assert ev["ticks_walked"].to_list() == [3, 0, 0, 0, 0]
    assert ev["qty"].to_list() == pytest.approx([4.0, 0.5, 1.0, 1.0, 1.0])
    assert ev["notional"][0] == pytest.approx(400.5)
    assert ev["vwap"].to_list() == pytest.approx([100.125, 100.3, 100.2, 100.1, 100.0])
    assert ev["is_buyer_maker"].to_list() == [False, False, True, True, True]
    assert ev["time"].to_list() == [ms(1000)] * 4 + [ms(2000)]


def test_attach_mid_same_timestamp_takes_higher_update_id():
    # Two quotes stamped 1000, given in reversed update_id order:
    #   update_id 20: 100.0 / 100.1 -> mid 100.05   (the later quote)
    #   update_id 10:  99.9 / 100.0 -> mid  99.95
    # The trade at 2000 must see the quote with the higher update_id, not the last row.
    quotes = pl.LazyFrame(
        {
            "time": [ms(1000), ms(1000)],
            "bid": [100.0, 99.9],
            "ask": [100.1, 100.0],
            "update_id": [20, 10],
        }
    )
    trades = pl.LazyFrame({"time": [ms(2000)], "price": [100.1], "is_buyer_maker": [False]})
    out = attach_mid(trades, quotes).collect()
    assert out["mid"].to_list() == pytest.approx([100.05])
    assert "update_id" not in out.columns
