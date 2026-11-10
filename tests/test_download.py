"""Downloader helpers on tiny synthetic CSVs (no network)."""

from datetime import date, datetime, timezone

import polars as pl

from crypto_microstructure.download import DATASETS, csv_to_parquet, file_url


def test_file_url_daily_and_monthly():
    day = date(2024, 3, 12)
    assert file_url(DATASETS["perp_bookTicker"], "BTCUSDT", day) == (
        "https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/"
        "BTCUSDT-bookTicker-2024-03-12.zip"
    )
    assert file_url(DATASETS["perp_fundingRate"], "BTCUSDT", day).endswith(
        "monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-03.zip"
    )


def test_bookticker_with_header_ms(tmp_path):
    csv = tmp_path / "book.csv"
    csv.write_text(
        "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,"
        "transaction_time,event_time\n"
        "1,100.0,2.5,100.1,1.0,1710201600000,1710201600005\n"
    )
    out = csv_to_parquet(csv, DATASETS["perp_bookTicker"], tmp_path / "book.parquet")
    df = pl.read_parquet(out)
    assert df.columns == list(DATASETS["perp_bookTicker"].columns)
    assert df["bid_price"][0] == 100.0
    assert df["transaction_time"][0] == datetime(2024, 3, 12, tzinfo=timezone.utc)
    assert df["event_time"][0] == datetime(2024, 3, 12, 0, 0, 0, 5000, tzinfo=timezone.utc)


def test_spot_trades_without_header_us(tmp_path):
    # Spot files have no header; from 2025 timestamps are in microseconds.
    csv = tmp_path / "trades.csv"
    csv.write_text("7,100.0,0.5,50.0,1710201600000001,True,True\n")
    out = csv_to_parquet(csv, DATASETS["spot_trades"], tmp_path / "trades.parquet")
    df = pl.read_parquet(out)
    assert df["trade_id"][0] == 7
    assert df["is_buyer_maker"][0] is True
    assert df["time"][0] == datetime(2024, 3, 12, 0, 0, 0, 1, tzinfo=timezone.utc)
