"""Turn a single VIX number into a usable weekly SPY implied-vol surface.

This is the biggest approximation in the free-data build. VIX is a 30-day,
variance-swap-style measure of SPX vol. We need per-strike IV for options
expiring in 1-7 days. Two adjustments:

1. Term structure. Short-dated ATM vol is usually a bit BELOW VIX when the
   curve is in contango (calm markets) and ABOVE it when inverted (stress).
   We proxy the regime with VIX relative to its own recent average.

2. Skew. Equity-index puts trade at higher IV than calls; the smile steepens
   for shorter tenors. We apply a moneyness-based skew in log-strike space,
   calibrated to typical SPY 1-week smile shape.

Every constant here is a documented rule-of-thumb, not a fitted parameter.
Replace this module with real quotes when you upgrade the data source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def atm_vol_from_vix(vix, vix_baseline=None):
    """Estimate 1-week ATM IV (decimal, e.g. 0.14) from spot VIX.

    ``vix`` and ``vix_baseline`` are in VIX points (e.g. 18.5). If a baseline
    (recent VIX average) is given, we tilt: calm regimes -> short vol under
    VIX; stressed regimes -> short vol over VIX.
    """
    vix = np.asarray(vix, dtype=float)
    v = vix / 100.0
    if vix_baseline is None:
        # Static contango assumption: ~1-week vol ~ 92% of 30-day vol.
        return v * 0.92
    base = np.asarray(vix_baseline, dtype=float) / 100.0
    ratio = np.clip(v / np.maximum(base, 1e-6), 0.5, 3.0)
    # ratio 1.0 -> factor 0.92 ; ratio 1.5 -> ~1.06 ; ratio 0.7 -> ~0.83
    factor = 0.92 + 0.28 * (ratio - 1.0)
    factor = np.clip(factor, 0.70, 1.60)
    return v * factor


def skew_adjust(atm_vol, strike, spot, dte):
    """Return per-strike IV given an ATM vol and a strike.

    Uses a simple parabolic smile in log-moneyness k = ln(K/S), with the
    downside (puts, k<0) steeper than the upside. Slope scales with 1/sqrt(T)
    so short-dated smiles are wider, matching observed SPY behaviour.
    """
    atm_vol = np.asarray(atm_vol, dtype=float)
    strike = np.asarray(strike, dtype=float)
    k = np.log(strike / float(spot))
    t = max(float(dte) / 365.0, 1.0 / 365.0)
    tenor_scale = np.sqrt(0.019 / t)  # 0.019 yr ~ 7 calendar days -> scale 1.0

    down_slope = 1.4 * tenor_scale   # put wing
    up_slope = 0.45 * tenor_scale    # call wing
    curv = 6.0 * tenor_scale         # smile curvature

    slope = np.where(k < 0, -down_slope, -up_slope)
    iv = atm_vol * (1.0 + slope * k + curv * k**2)
    return np.clip(iv, 0.03, 3.0)


def build_surface(spot, vix, dte, strikes, vix_baseline=None):
    """Convenience: DataFrame of (strike, moneyness, iv) for one expiry."""
    atm = float(atm_vol_from_vix(vix, vix_baseline))
    strikes = np.asarray(strikes, dtype=float)
    iv = skew_adjust(atm, strikes, spot, dte)
    return pd.DataFrame({
        "strike": strikes,
        "moneyness": strikes / spot,
        "iv": iv,
        "atm_vol": atm,
    })


def expected_move(spot, atm_vol, dte, basis="calendar"):
    """1-sigma expected move in price terms over ``dte`` days."""
    days = 365.0 if basis == "calendar" else 252.0
    dte = np.maximum(np.asarray(dte, dtype=float), 0.0)
    return float(spot) * np.asarray(atm_vol, dtype=float) * np.sqrt(dte / days)
