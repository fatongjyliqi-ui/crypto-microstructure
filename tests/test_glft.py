"""GLFT quotes against hand-computed cases and known limits."""

import math

import numpy as np
import pytest

from crypto_microstructure.glft import (
    eta,
    glft_quotes,
    glft_spread_approx,
    glft_spread_exact,
)

LN2 = math.log(2)
# gamma = kappa = 1, A = 4, sigma = sqrt(2):
#   alpha = kappa gamma sigma^2 / 2 = 1,  eta = 4 * (1 + 1)^(-2) = 1
HAND = dict(gamma=1.0, kappa=1.0, A=4.0, sigma=math.sqrt(2))


def test_eta_hand():
    assert eta(1.0, 1.0, 4.0) == pytest.approx(1.0)


def test_stationary_quotes_q1_hand_computed():
    # Q = 1: M = [[1, -1, 0], [-1, 0, -1], [0, -1, 1]], eigenvalues -1, 1, 2.
    # Smallest: -1 with eigenvector (1, 2, 1), so v_0 / v_{+-1} = 2.
    #   delta_b(0) = log 2 + log(v_0 / v_1) = 2 log 2 = delta_a(0)  -> spread 4 log 2
    #   delta_b(-1) = log 2 + log(v_-1 / v_0) = 0 = delta_a(1)
    d = glft_quotes(Q=1, **HAND)
    assert d["delta_b"][:2] == pytest.approx([0.0, 2 * LN2], abs=1e-12)
    assert d["delta_a"][1:] == pytest.approx([2 * LN2, 0.0], abs=1e-12)
    assert math.isnan(d["delta_b"][2]) and math.isnan(d["delta_a"][0])  # bounds q = +-Q
    assert glft_spread_exact(Q=1, **HAND) == pytest.approx(4 * LN2)


def test_at_maturity_spread_is_risk_term_only():
    # tau = 0: v = 1, so every quote is (1/gamma) log(1 + gamma/kappa) = log 2
    assert glft_spread_exact(Q=3, tau=0.0, **HAND) == pytest.approx(2 * LN2)


def test_finite_horizon_converges_to_stationary():
    stationary = glft_spread_exact(Q=3, **HAND)
    assert glft_spread_exact(Q=3, tau=50.0, **HAND) == pytest.approx(stationary, rel=1e-10)
    assert glft_spread_exact(Q=3, tau=0.5, **HAND) < stationary


def test_quotes_symmetric_in_inventory():
    d = glft_quotes(gamma=0.3, kappa=2.0, A=5.0, sigma=1.5, Q=6)
    assert d["delta_b"][:-1] == pytest.approx(d["delta_a"][1:][::-1])


def test_approx_hand_computed():
    # gamma = kappa = A = 1, sigma = sqrt(2):
    #   2 log 2 + sqrt(1 * 2 / (2 * 1 * 1) * 2^2) = 2 log 2 + 2
    assert glft_spread_approx(1.0, 1.0, 1.0, math.sqrt(2)) == pytest.approx(2 * LN2 + 2)


def test_risk_neutral_limit_approx_is_two_over_kappa():
    # Inventory term sqrt(gamma sigma^2 e / (2 kappa A)) ~ 4.3e-6 at gamma = 1e-9.
    assert glft_spread_approx(1e-9, 4.0, 9.0, 0.7) == pytest.approx(0.5, rel=1e-4)


@pytest.mark.parametrize("Q", [1, 3, 20])
def test_risk_neutral_limit_exact_with_inventory_bound(Q):
    # gamma -> 0: v is the Perron vector of the path graph, v_0 / v_1 = 1 / cos(pi/(2Q+2)),
    # so the spread is 2/kappa + (2/kappa) log(1 / cos(pi / (2Q + 2))).
    # Q = 1: 0.5 + 0.5 log(sqrt 2) = 0.5 + 0.25 log 2 = 0.67329 for kappa = 4.
    kappa = 4.0
    expected = 2 / kappa + 2 / kappa * math.log(1 / math.cos(math.pi / (2 * Q + 2)))
    got = glft_spread_exact(1e-9, kappa, 9.0, 0.7, Q=Q)
    assert got == pytest.approx(expected, rel=1e-6)
    if Q == 1:
        assert got == pytest.approx(0.5 + 0.25 * LN2, rel=1e-6)


def test_approx_close_to_exact_when_inventory_term_small():
    # XRP-like inputs (ticks, per second): the Gaussian approximation is accurate here.
    kappa, A, sigma = 4.086, 9.073, 0.738
    for r in [1e-3, 1e-2]:
        ex = glft_spread_exact(r * kappa, kappa, A, sigma, Q=50)
        assert glft_spread_approx(r * kappa, kappa, A, sigma) == pytest.approx(ex, rel=2e-3)


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        glft_spread_approx(0.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        glft_spread_exact(1.0, 1.0, 1.0, 1.0, Q=0)


def test_spread_increases_with_gamma():
    s = [glft_spread_exact(g, 2.0, 5.0, 1.0, Q=30) for g in (0.01, 0.1, 1.0)]
    assert np.all(np.diff(s) > 0)
