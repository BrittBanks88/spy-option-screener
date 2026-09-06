"""Black-Scholes-Merton pricing and Greeks for European options.

SPY pays dividends, so we use the continuous-dividend-yield form. All rates
are continuously compounded and annualized. Time ``t`` is in years.

The functions are vectorized: pass scalars or numpy arrays for S, K, t, sigma.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Trading-day convention for converting calendar days -> year fraction.
# 252 trading days is standard for equity-index options.
TRADING_DAYS = 252.0


def _d1_d2(S, K, t, sigma, r, q):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    # Guard against t == 0 or sigma == 0 producing divide-by-zero.
    vol_sqrt_t = np.maximum(sigma * np.sqrt(np.maximum(t, 0.0)), 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def price(S, K, t, sigma, r=0.04, q=0.013, kind="call"):
    """Option price. ``kind`` is 'call' or 'put'.

    Defaults: r ~ short-term T-bill, q ~ SPY trailing dividend yield.
    """
    d1, d2 = _d1_d2(S, K, t, sigma, r, q)
    disc_r = np.exp(-r * np.asarray(t, dtype=float))
    disc_q = np.exp(-q * np.asarray(t, dtype=float))
    if kind == "call":
        val = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif kind == "put":
        val = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    # At expiry, collapse to intrinsic value.
    t_arr = np.asarray(t, dtype=float)
    intrinsic = np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)
    val = np.where(t_arr <= 0.0, intrinsic, val)
    return val


def greeks(S, K, t, sigma, r=0.04, q=0.013, kind="call"):
    """Return a dict of delta, gamma, theta (per day), vega (per 1 vol pt), rho.

    theta is expressed per calendar day; vega per 1 percentage-point IV move.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    t = np.asarray(t, dtype=float)
    d1, d2 = _d1_d2(S, K, t, sigma, r, q)
    disc_r = np.exp(-r * t)
    disc_q = np.exp(-q * t)
    pdf_d1 = norm.pdf(d1)
    sqrt_t = np.sqrt(np.maximum(t, 1e-12))

    if kind == "call":
        delta = disc_q * norm.cdf(d1)
        theta_yr = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * t * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta_yr = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * t * disc_r * norm.cdf(-d2)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t  # per 1.0 change in sigma

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta_yr / 365.0,   # per calendar day
        "vega": vega / 100.0,        # per 1 vol point (0.01)
        "rho": rho / 100.0,
    }


def implied_vol(target_price, S, K, t, r=0.04, q=0.013, kind="call",
                lo=1e-4, hi=5.0, tol=1e-6, max_iter=100):
    """Invert Black-Scholes for sigma via bisection. Robust, no derivative needed.

    Returns np.nan if the target price is outside the no-arbitrage bounds.
    """
    S, K, t = float(S), float(K), float(t)
    intrinsic = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    upper = S if kind == "call" else K
    if target_price <= intrinsic - tol or target_price >= upper:
        return np.nan

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = float(price(S, K, t, mid, r, q, kind))
        if abs(val - target_price) < tol:
            return mid
        if val > target_price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def years_from_days(calendar_days, basis="calendar"):
    """Convert a day count to a year fraction.

    basis='calendar' uses 365; basis='trading' uses 252. Short-DTE option
    decay tracks trading days more closely, but quotes use calendar time.
    """
    if basis == "trading":
        return np.asarray(calendar_days, dtype=float) / TRADING_DAYS
    return np.asarray(calendar_days, dtype=float) / 365.0
