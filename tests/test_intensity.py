"""Fill-intensity estimation on hand-computed and synthetic geometric data."""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from crypto_microstructure.intensity import fit_intensity, geometric_theta, ticks_beyond_touch

T0 = datetime(2024, 3, 1, tzinfo=timezone.utc)


def test_geometric_theta_hand_computed():
    x = np.array([0, 0, 0, 1, 4])
    # Uncensored: Xbar = 5/5 = 1 -> theta = 1 / (1 + 1) = 0.5
    assert geometric_theta(x) == pytest.approx(0.5)
    # Censored at K = 2: min(x, 2) = 0,0,0,1,2 -> S = 3; N_<2 = 4 -> theta = 3 / 7
    assert geometric_theta(x, k_max=2) == pytest.approx(3 / 7)


@pytest.mark.parametrize("k_max", [None, 3])
def test_geometric_theta_recovers_synthetic_theta(k_max):
    # X = G - 1 with G ~ Geometric(p) on {1, 2, ...} has P(X >= k) = (1 - p)^k.
    theta = 0.3
    x = np.random.default_rng(7).geometric(1 - theta, size=200_000) - 1
    # SE of theta_hat is about theta (1 - theta) / sqrt(n * theta) ~ 0.0009 here.
    assert geometric_theta(x, k_max) == pytest.approx(theta, abs=0.005)


def test_geometric_theta_rejects_negative():
    with pytest.raises(ValueError):
        geometric_theta(np.array([0, -1]))


def test_ticks_beyond_touch():
    # Buy taker, ask 100.1, worst fill 100.3 -> 2 ticks beyond; spread 100.1 - 100.0 = 1 tick.
    # Sell taker, bid 100.0, worst fill 99.9 -> 1 tick beyond.
    # Buy taker, ask 100.1, worst fill 100.0 -> -1 (inside the touch); spread 2 ticks.
    ev = pl.LazyFrame(
        {
            "is_buyer_maker": [False, True, False],
            "worst_price": [100.3, 99.9, 100.0],
            "bid": [100.0, 100.0, 99.9],
            "ask": [100.1, 100.1, 100.1],
        }
    )
    out = ticks_beyond_touch(ev, tick=0.1).collect()
    assert out["x"].to_list() == [2, 1, -1]
    assert out["spread_ticks"].to_list() == [1, 1, 2]


def test_fit_intensity_hand_computed():
    # All events in one 5-minute block, exposure 10 s.
    #   buy:  x = 0, 0, 1 -> Xbar = 1/3 -> theta = (1/3) / (4/3) = 1/4, kappa = ln 4
    #         Lambda = 3 / 10 = 0.3; A_touched = 0.3 / sqrt(1/4) = 0.6, A_through = 0.15
    #   sell: x = 0, 2    -> Xbar = 1   -> theta = 1/2, Lambda = 0.2
    #   both: x = 0,0,1,0,2 -> S = 3, n = 5 -> theta = 3/8; Lambda = 5 / (2 * 10) = 0.25
    # Excluded: a buy with a 2-tick spread and a sell inside the touch (x = -1).
    ev = pl.DataFrame(
        {
            "time": [T0 + timedelta(seconds=i) for i in range(7)],
            "is_buyer_maker": [False, False, False, True, True, False, True],
            "spread_ticks": [1, 1, 1, 1, 1, 2, 1],
            "x": [0, 0, 1, 0, 2, 3, -1],
        }
    )
    fit = fit_intensity(ev, exposure_s=10.0, n_boot=50)
    out = {r["side"]: r for r in fit.iter_rows(named=True)}
    buy, sell, both = out["buy"], out["sell"], out["both"]
    assert buy["n_events"] == 3 and buy["n_beyond_touch"] == 1
    assert buy["theta"] == pytest.approx(0.25)
    assert buy["kappa"] == pytest.approx(math.log(4))
    assert buy["lambda_per_s"] == pytest.approx(0.3)
    assert buy["a_touched"] == pytest.approx(0.6)
    assert buy["a_through"] == pytest.approx(0.15)
    assert buy["two_over_kappa"] == pytest.approx(2 / math.log(4))
    assert sell["theta"] == pytest.approx(0.5) and sell["lambda_per_s"] == pytest.approx(0.2)
    assert both["theta"] == pytest.approx(3 / 8) and both["lambda_per_s"] == pytest.approx(0.25)
    # One block only: every bootstrap resample is the full sample, so the CI collapses.
    for r in (buy, sell, both):
        assert r["kappa_lo"] == pytest.approx(r["kappa"])
        assert r["kappa_hi"] == pytest.approx(r["kappa"])


def test_fit_intensity_no_event_beyond_touch():
    # Only touch fills: theta = 0, kappa = inf, 2/kappa = 0 (not identified from above).
    ev = pl.DataFrame(
        {
            "time": [T0, T0 + timedelta(seconds=1)],
            "is_buyer_maker": [False, True],
            "spread_ticks": [1, 1],
            "x": [0, 0],
        }
    )
    both = fit_intensity(ev, exposure_s=1.0, n_boot=20).filter(pl.col("side") == "both").row(
        0, named=True
    )
    assert both["theta"] == 0 and math.isinf(both["kappa"]) and both["two_over_kappa"] == 0
