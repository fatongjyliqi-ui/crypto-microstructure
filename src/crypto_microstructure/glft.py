"""Optimal market-making quotes of Gueant, Lehalle & Fernandez-Tapia (2013).

Model: mid S_t = S_0 + sigma W_t; the market maker quotes S - delta_b and S + delta_a and
is filled at rates A exp(-kappa delta) per side, one unit per fill; inventory q stays in
{-Q, ..., Q}; objective E[-exp(-gamma (X_T + q_T S_T))].

With u(t, x, q, s) = -exp(-gamma (x + q s)) v_q(t)^(-gamma / kappa) the HJB equation
becomes the linear ODE system
    v_q'(t) = alpha q^2 v_q - eta (v_{q+1} 1{q < Q} + v_{q-1} 1{q > -Q}),  v_q(T) = 1,
    alpha = kappa gamma sigma^2 / 2,   eta = A (1 + gamma / kappa)^(-(1 + kappa / gamma)),
so v(t) = exp(-M (T - t)) 1 with M symmetric tridiagonal (diagonal alpha q^2, off-diagonal
-eta). The optimal quotes are
    delta_b(q) = (1/gamma) log(1 + gamma/kappa) + (1/kappa) log(v_q / v_{q+1})   (q < Q)
    delta_a(q) = (1/gamma) log(1 + gamma/kappa) + (1/kappa) log(v_q / v_{q-1})   (q > -Q)
For T - t -> inf, v is proportional to the eigenvector of the smallest eigenvalue of M
(positive by Perron-Frobenius), so the quotes become stationary.

Closed-form approximation (q treated as continuous, Gaussian eigenfunction): the spread
does not depend on q,
    delta_a + delta_b ~ (2/gamma) log(1 + gamma/kappa)
                        + sqrt(gamma sigma^2 / (2 kappa A) (1 + gamma/kappa)^(1 + kappa/gamma)).
As gamma -> 0 the approximation tends to 2 / kappa (the correction is O(sqrt(gamma))).
The exact stationary spread with a finite bound Q does not: alpha -> 0 and M -> -eta times
the adjacency matrix of a path with 2Q + 1 nodes, whose Perron eigenvector is
sin(pi (j + 1) / (2Q + 2)), so v_0 / v_1 = 1 / cos(pi / (2Q + 2)) and the limit is
    2/kappa + (2/kappa) log(1 / cos(pi / (2Q + 2)))  ~  2/kappa (1 + pi^2 / (8 (Q + 1)^2)):
avoiding the inventory bound widens the quotes even for a risk-neutral market maker.

Units: delta, sigma, 1/kappa in ticks (sigma in ticks per sqrt(second)), A per second,
gamma per tick and unit of inventory.
"""

from __future__ import annotations

import math

import numpy as np


def _check(gamma: float, kappa: float, A: float, sigma: float) -> None:
    if not (gamma > 0 and kappa > 0 and A > 0 and sigma >= 0):
        raise ValueError("need gamma > 0, kappa > 0, A > 0, sigma >= 0")


def eta(gamma: float, kappa: float, A: float) -> float:
    """A (1 + gamma/kappa)^(-(1 + kappa/gamma)), computed in logs for small gamma."""
    return A * math.exp(-(1 + kappa / gamma) * math.log1p(gamma / kappa))


def glft_spread_approx(gamma: float, kappa: float, A: float, sigma: float) -> float:
    """Closed-form approximate spread delta_a + delta_b (independent of q)."""
    _check(gamma, kappa, A, sigma)
    base = 2 / gamma * math.log1p(gamma / kappa)
    log_pow = (1 + kappa / gamma) * math.log1p(gamma / kappa)
    return base + math.sqrt(gamma * sigma**2 / (2 * kappa * A) * math.exp(log_pow))


def glft_v(
    gamma: float, kappa: float, A: float, sigma: float, Q: int, tau: float | None = None
) -> np.ndarray:
    """v_q for q = -Q..Q, up to a positive factor (only ratios enter the quotes).

    tau = T - t; None means the stationary limit tau -> inf.
    """
    _check(gamma, kappa, A, sigma)
    q = np.arange(-Q, Q + 1)
    alpha = kappa * gamma * sigma**2 / 2
    M = np.diag(alpha * q.astype(float) ** 2)
    off = -eta(gamma, kappa, A) * np.ones(2 * Q)
    M += np.diag(off, 1) + np.diag(off, -1)
    lam, U = np.linalg.eigh(M)  # ascending eigenvalues
    if tau is None:
        return np.abs(U[:, 0])
    # exp(-M tau) 1 = U diag(exp(-lam tau)) U^T 1; factor out exp(-lam_min tau) against underflow
    return U @ (np.exp(-(lam - lam[0]) * tau) * (U.T @ np.ones(2 * Q + 1)))


def glft_quotes(
    gamma: float, kappa: float, A: float, sigma: float, Q: int, tau: float | None = None
) -> dict[str, np.ndarray]:
    """Optimal delta_b and delta_a for q = -Q..Q (NaN where that side is not quoted).

    For large sigma^2 / A the stationary v_q is a very narrow bump and underflows to 0 for
    large |q|; quotes there come out as inf or NaN. Quotes near q = 0 are unaffected.
    """
    v = glft_v(gamma, kappa, A, sigma, Q, tau)
    base = math.log1p(gamma / kappa) / gamma
    delta_b = np.full(2 * Q + 1, np.nan)
    delta_a = np.full(2 * Q + 1, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_b[:-1] = base + np.log(v[:-1] / v[1:]) / kappa
        delta_a[1:] = base + np.log(v[1:] / v[:-1]) / kappa
    return {"q": np.arange(-Q, Q + 1), "delta_b": delta_b, "delta_a": delta_a}


def glft_spread_exact(
    gamma: float, kappa: float, A: float, sigma: float, Q: int, tau: float | None = None
) -> float:
    """delta_a + delta_b at q = 0 from the ODE system (Q >= 1)."""
    if Q < 1:
        raise ValueError("Q >= 1 needed to quote both sides at q = 0")
    d = glft_quotes(gamma, kappa, A, sigma, Q, tau)
    return float(d["delta_a"][Q] + d["delta_b"][Q])
