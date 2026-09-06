"""Build an option chain -- synthetic (for backtests) or real (for live use)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..pricing import black_scholes as bs
from ..pricing import vol_model as vm


def synthetic_chain(spot, vix, dte, vix_baseline=None, width=0.08, step=1.0,
                    r=0.04, q=0.013) -> pd.DataFrame:
    """Reconstruct a weekly chain around ``spot`` from VIX.

    ``width`` is the +/- fraction of spot to span with strikes; ``step`` the
    strike increment in dollars (SPY lists $1 strikes near the money).
    Returns calls and puts with model price, IV and Greeks.
    """
    lo = np.floor(spot * (1 - width))
    hi = np.ceil(spot * (1 + width))
    strikes = np.arange(lo, hi + step, step)
    t = bs.years_from_days(dte, basis="calendar")

    surf = vm.build_surface(spot, vix, dte, strikes, vix_baseline)
    rows = []
    for kind in ("call", "put"):
        iv = surf["iv"].values
        px = bs.price(spot, strikes, t, iv, r, q, kind)
        g = bs.greeks(spot, strikes, t, iv, r, q, kind)
        rows.append(pd.DataFrame({
            "type": kind,
            "strike": strikes,
            "dte": dte,
            "iv": iv,
            "mid": px,
            "delta": g["delta"],
            "gamma": g["gamma"],
            "theta": g["theta"],
            "vega": g["vega"],
            "moneyness": strikes / spot,
        }))
    chain = pd.concat(rows, ignore_index=True)
    chain.attrs["spot"] = float(spot)
    chain.attrs["atm_vol"] = float(surf["atm_vol"].iloc[0])
    return chain


def enrich_live_chain(chain: pd.DataFrame, r=0.04, q=0.013,
                      fallback_vix: float | None = None) -> pd.DataFrame:
    """Add model Greeks to a yfinance live chain, cleaning bad quotes.

    yfinance option data is unreliable outside regular trading hours and for
    ~0-DTE strikes: zero bid/ask, missing IV, prices stuck at intrinsic. We
    drop rows we can't trust rather than let them poison the ranking.
    """
    spot = float(chain.attrs["spot"])
    out = chain.copy()
    out["dte"] = out["dte"].clip(lower=0)
    t = np.maximum(bs.years_from_days(out["dte"].values, basis="calendar"),
                   0.5 / 365.0)

    # Drop quotes with no real market or price <= intrinsic (no time value).
    intrinsic = np.where(out["type"].eq("call"),
                         np.maximum(spot - out["strike"], 0.0),
                         np.maximum(out["strike"] - spot, 0.0))
    tradable = (out["bid"].fillna(0) > 0) & (out["ask"].fillna(0) > 0)
    has_tv = out["mid"] > intrinsic + 0.02
    out = out[tradable & has_tv].copy()
    if out.empty:
        raise RuntimeError("No tradable live quotes (market closed or stale). "
                           "Use 'Reconstructed from VIX' instead.")
    t = np.maximum(bs.years_from_days(out["dte"].values, basis="calendar"),
                   0.5 / 365.0)
    intrinsic = np.where(out["type"].eq("call"),
                         np.maximum(spot - out["strike"], 0.0),
                         np.maximum(out["strike"] - spot, 0.0))

    # Trust the exchange IV only if it's sane; else solve from mid.
    iv = out["iv"].astype(float).values.copy()
    bad = ~np.isfinite(iv) | (iv <= 0.01) | (iv > 3.0)
    for pos, i in enumerate(np.where(bad)[0]):
        row = out.iloc[i]
        solved = bs.implied_vol(float(row["mid"]), spot, float(row["strike"]),
                                float(t[i]), r, q, row["type"])
        iv[i] = solved
    out["iv"] = iv
    out = out[np.isfinite(out["iv"]) & (out["iv"] > 0.01)].copy()
    if out.empty:
        raise RuntimeError("Could not derive IV from any live quote.")

    t = np.maximum(bs.years_from_days(out["dte"].values, basis="calendar"),
                   0.5 / 365.0)
    for kind in ("call", "put"):
        m = (out["type"] == kind).values
        if not m.any():
            continue
        gg = bs.greeks(spot, out.loc[m, "strike"].values, t[m],
                       out.loc[m, "iv"].values, r, q, kind)
        for key in ("delta", "gamma", "theta", "vega"):
            out.loc[m, key] = gg[key]

    out["moneyness"] = out["strike"] / spot
    out["spread_pct"] = ((out["ask"] - out["bid"])
                         / out["mid"].replace(0, np.nan)).clip(0, 2)
    atm = out.loc[(out["moneyness"] - 1).abs() < 0.015, "iv"]
    out.attrs["spot"] = spot
    out.attrs["atm_vol"] = float(atm.median()) if len(atm) else float(out["iv"].median())
    return out.reset_index(drop=True)
