"""Mid / quoted spread on a tiny synthetic book with hand-computed values."""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from crypto_microstructure.spread import add_mid_and_quoted_spread, sample_on_grid

T0 = datetime(2024, 3, 12, tzinfo=timezone.utc)


def t(sec: float) -> datetime:
    """Timestamp `sec` seconds after midnight of the test day."""
    return T0 + timedelta(seconds=sec)


@pytest.fixture
def book() -> pl.LazyFrame:
    # Three quotes. Hand-computed:
    #   100.0 / 100.2 -> mid 100.1, spread 0.2, bps = 0.2 / 100.1 * 1e4 = 19.98002
    #    99.9 / 100.1 -> mid 100.0, spread 0.2, bps = 20.0
    #   100.0 / 100.5 -> mid 100.25, spread 0.5, bps = 0.5 / 100.25 * 1e4 = 49.87531
    return pl.LazyFrame(
        {
            "bid_price": [100.0, 99.9, 100.0],
            "ask_price": [100.2, 100.1, 100.5],
            "transaction_time": [t(0.5), t(1.2), t(1.7)],
        }
    )


def test_mid_and_quoted_spread(book):
    out = add_mid_and_quoted_spread(book).collect()
    assert out["mid"].to_list() == pytest.approx([100.1, 100.0, 100.25])
    assert out["spread"].to_list() == pytest.approx([0.2, 0.2, 0.5])
    assert out["spread_bps"].to_list() == pytest.approx([19.98002, 20.0, 49.87531], rel=1e-6)


def test_sample_on_grid_takes_last_quote_at_or_before_each_point(book):
    # Grid 0s, 1s, 2s:
    #   t=0: no quote yet          -> null
    #   t=1: last quote is t=0.5   -> bid 100.0
    #   t=2: last quote is t=1.7   -> bid 100.0, ask 100.5 (the t=1.2 quote is skipped)
    out = sample_on_grid(book, t(0), t(2), every="1s").collect()
    assert out["bid_price"].to_list() == [None, 100.0, 100.0]
    assert out["ask_price"].to_list() == [None, 100.2, 100.5]


def test_time_weighted_mean_differs_from_event_mean(book):
    # Events: mean spread = (0.2 + 0.2 + 0.5) / 3 = 0.3.
    # Grid at 1s, 2s (seconds after the first quote): spreads 0.2 and 0.5 -> mean 0.35.
    spread = add_mid_and_quoted_spread(book)
    event_mean = spread.select(pl.col("spread").mean()).collect().item()
    grid_mean = (
        sample_on_grid(spread, t(1), t(2)).select(pl.col("spread").mean()).collect().item()
    )
    assert event_mean == pytest.approx(0.3)
    assert grid_mean == pytest.approx(0.35)
