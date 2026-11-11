from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from crypto_microstructure.load import load_perp, on_grid

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
    day = load_perp(DAY, symbol="XRPUSDT", data_dir=tmp_path)
    assert day.tick == 0.0001
    assert day.n_off_grid_dropped == 1
    assert day.quotes.collect()["bid"].to_list() == [0.5934, 0.5935]
    assert day.trades.collect()["off_grid"].to_list() == [False, True]


def test_load_perp_btc_tick_would_reject_every_xrp_quote(tmp_path):
    # Guards against silently using the BTCUSDT default tick for an alt.
    write_day(tmp_path, "XRPUSDT", bids=[0.5934], asks=[0.5935], trade_prices=[0.5935])
    day = load_perp(DAY, symbol="XRPUSDT", tick=0.1, data_dir=tmp_path)
    assert day.n_off_grid_dropped == 1


def test_load_perp_unknown_symbol_needs_tick(tmp_path):
    with pytest.raises(ValueError, match="pass tick"):
        load_perp(DAY, symbol="FOOUSDT", data_dir=tmp_path)
