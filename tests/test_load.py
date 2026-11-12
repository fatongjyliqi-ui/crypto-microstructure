import math
import warnings
from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from crypto_microstructure.load import load_perp, observed_tick, on_grid

DAY = date(2024, 3, 1)
T0 = datetime(2024, 3, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "price, tick, expected",
    [
        (61431.9, 0.1, True),
        (61431.87, 0.1, False),  # real off-grid BTCUSDT ask from 2024-03-01
        (0.5935, 0.0001, True),
        (0.59355, 0.0001, False),  # half a tick
        (0.12614, 0.00001, True),
        (42.253, 0.001, True),
        (42.2535, 0.001, False),
    ],
)
def test_on_grid(price, tick, expected):
    out = pl.DataFrame({"p": [price]}).select(on_grid("p", tick)).item()
    assert out is expected


def write_day(root, symbol, bids, asks, trade_prices):
    """Minimal bookTicker and aggTrades Parquet files for one symbol."""
    book_dir = root / "parquet" / "perp_bookTicker"
    agg_dir = root / "parquet" / "perp_aggTrades"
    book_dir.mkdir(parents=True)
    agg_dir.mkdir(parents=True)
    n = len(bids)
    pl.DataFrame(
        {
            "update_id": list(range(n)),
            "bid_price": bids,
            "bid_qty": [1.0] * n,
            "ask_price": asks,
            "ask_qty": [1.0] * n,
            "transaction_time": [T0 + timedelta(seconds=i) for i in range(n)],
            "event_time": [T0 + timedelta(seconds=i) for i in range(n)],
        }
    ).write_parquet(book_dir / f"{symbol}-bookTicker-{DAY}.parquet")
    m = len(trade_prices)
    pl.DataFrame(
        {
            "agg_trade_id": list(range(m)),
            "price": trade_prices,
            "qty": [1.0] * m,
            "first_trade_id": list(range(m)),
            "last_trade_id": list(range(m)),
            "transact_time": [T0 + timedelta(seconds=i) for i in range(m)],
            "is_buyer_maker": [False] * m,
        }
    ).write_parquet(agg_dir / f"{symbol}-aggTrades-{DAY}.parquet")


def test_load_perp_uses_symbol_tick(tmp_path):
    # Quote 1 has a half-tick bid (0.59355) and is dropped; trade 2 is off-grid and flagged.
    write_day(
        tmp_path,
        "XRPUSDT",
        bids=[0.5934, 0.59355, 0.5935],
        asks=[0.5935, 0.5936, 0.5936],
        trade_prices=[0.5935, 0.59355],
    )
    # 1 of 3 quotes off the grid (33%) is above MAX_OFF_GRID_SHARE, so it also warns.
    with pytest.warns(UserWarning, match="may be too large"):
        day = load_perp(DAY, symbol="XRPUSDT", data_dir=tmp_path)
    assert day.tick == 0.0001
    # Remaining levels 0.5934, 0.5935, 0.5936 -> gaps 1, 1 ticks -> observed 0.0001
    assert day.observed_tick == pytest.approx(0.0001)
    assert day.n_off_grid_dropped == 1
    assert day.quotes.collect()["bid"].to_list() == [0.5934, 0.5935]
    assert day.trades.collect()["off_grid"].to_list() == [False, True]


def test_load_perp_btc_tick_would_reject_every_xrp_quote(tmp_path):
    # Guards against silently using the BTCUSDT default tick for an alt.
    write_day(tmp_path, "XRPUSDT", bids=[0.5934], asks=[0.5935], trade_prices=[0.5935])
    with pytest.warns(UserWarning, match="may be too large"):
        day = load_perp(DAY, symbol="XRPUSDT", tick=0.1, data_dir=tmp_path)
    assert day.n_off_grid_dropped == 1
    assert math.isnan(day.observed_tick)  # no valid quote left


def test_load_perp_unknown_symbol_needs_tick(tmp_path):
    with pytest.raises(ValueError, match="pass tick"):
        load_perp(DAY, symbol="FOOUSDT", data_dir=tmp_path)


@pytest.mark.parametrize(
    "prices, expected",
    [
        # BTC: 61431.9 - 61431.7 = 2 ticks, 61432.4 - 61431.9 = 5 ticks -> gcd 1 tick
        ([61431.7, 61431.9, 61432.4], 0.1),
        # every gap a multiple of 0.2 (2 and 6 tenths) -> 0.2, not 0.1
        ([100.0, 100.2, 100.8], 0.2),
        # XRP: gaps 2 and 5 in units of 0.0001 -> 0.0001
        ([0.5934, 0.5936, 0.5941], 0.0001),
        # DOGE: gaps 3 and 7 in units of 0.00001; unsorted input with a duplicate
        ([0.12624, 0.12614, 0.12617, 0.12617], 0.00001),
        # BTC with the real off-grid ask: 61431.87 - 61431.7 = 17 hundredths -> 0.01
        ([61431.7, 61431.87, 61431.9], 0.01),
    ],
)
def test_observed_tick(prices, expected):
    assert observed_tick(pl.Series(prices)) == pytest.approx(expected)


def test_observed_tick_needs_two_levels():
    assert math.isnan(observed_tick(pl.Series([0.5934, 0.5934])))


def test_load_perp_warns_if_tick_too_small(tmp_path):
    # All prices on the 0.0001 grid; assuming 0.00001 passes on_grid but the quotes only
    # use multiples of 0.0001 -> observed tick 10x the assumed one.
    write_day(tmp_path, "XRPUSDT", bids=[0.5934, 0.5935], asks=[0.5935, 0.5937],
              trade_prices=[0.5935])
    with pytest.warns(UserWarning, match="may be too small"):
        day = load_perp(DAY, symbol="XRPUSDT", tick=0.00001, data_dir=tmp_path)
    assert day.observed_tick == pytest.approx(0.0001)
    assert day.n_off_grid_dropped == 0


def test_load_perp_no_warning_on_clean_day(tmp_path):
    write_day(tmp_path, "XRPUSDT", bids=[0.5934, 0.5935], asks=[0.5935, 0.5937],
              trade_prices=[0.5935])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        day = load_perp(DAY, symbol="XRPUSDT", data_dir=tmp_path)
    assert day.observed_tick == pytest.approx(0.0001)
