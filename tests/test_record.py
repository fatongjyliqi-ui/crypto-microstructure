"""Live recorder parsing and buffering, using captured message shapes (no network)."""

from datetime import datetime, timezone

import polars as pl

from crypto_microstructure.record import ParquetBuffer, parse_message, rows_to_frame

UTC = timezone.utc

# Shapes as received on 2026-11-10. Spot has no timestamps; perp has T and E (ms).
SPOT_MSG = {"u": 1, "s": "BTCUSDT", "b": "100.00", "B": "2.5", "a": "100.01", "A": "0.5"}
PERP_MSG = {
    "e": "bookTicker", "u": 2, "s": "BTCUSDT", "ps": "BTCUSDT",
    "b": "100.0", "B": "1.0", "a": "100.1", "A": "3.0",
    "T": 1710201600000, "E": 1710201600005, "st": 1,
}  # fmt: skip

LOCAL_US = 1710201600012345  # 2024-03-12 00:00:00.012345 UTC


def test_parse_spot_has_no_exchange_times():
    row = parse_message(SPOT_MSG, LOCAL_US, session=1)
    assert row == (1, 100.0, 2.5, 100.01, 0.5, None, None, LOCAL_US, 1)


def test_rows_to_frame_types_and_times():
    df = rows_to_frame([parse_message(PERP_MSG, LOCAL_US, session=3)])
    assert df.columns[:5] == ["update_id", "bid_price", "bid_qty", "ask_price", "ask_qty"]
    assert df["ask_price"][0] == 100.1
    assert df["transaction_time"][0] == datetime(2024, 3, 12, tzinfo=UTC)
    assert df["event_time"][0] == datetime(2024, 3, 12, 0, 0, 0, 5000, tzinfo=UTC)
    assert df["local_time"][0] == datetime(2024, 3, 12, 0, 0, 0, 12345, tzinfo=UTC)
    assert df["session"][0] == 3


def test_buffer_splits_files_at_utc_midnight(tmp_path):
    buf = ParquetBuffer(tmp_path)
    before = 1710201599_900000  # 2024-03-11 23:59:59.9
    after = 1710201600_100000  # 2024-03-12 00:00:00.1
    buf.add(parse_message(SPOT_MSG, before, 1))
    buf.add(parse_message(SPOT_MSG, after, 1))  # new day -> previous day flushed
    buf.flush()
    assert (tmp_path / "2024-03-11" / f"part-{before}.parquet").exists()
    assert (tmp_path / "2024-03-12" / f"part-{after}.parquet").exists()
    assert pl.read_parquet(tmp_path / "*" / "*.parquet").height == 2
    assert buf.flush() is None  # empty buffer writes nothing
