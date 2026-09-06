"""Build an option chain -- synthetic (for backtests) or real (for live use)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..pricing import black_scholes as bs
from ..pricing import vol_model as vm
from ..data import polygon as _poly


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
    chain.attrs["source"] = "reconstructed"
    return chain


def prep_polygon_chain(chain: pd.DataFrame, r=0.04, q=0.013) -> pd.DataFrame:
    """A Polygon snapshot already carries IV + Greeks. Fill any gaps with the
    model, add mid/spread/moneyness, and keep the same columns score_chain wants.
    """
    spot = float(chain.attrs["spot"])
    out = chain.copy()
    out["moneyness"] = out["strike"] / spot
    if {"bid", "ask"}.issubset(out.columns):
        out["spread_pct"] = ((out["ask"] - out["bid"])
                             / out["mid"].replace(0, np.nan)).clip(0, 2)
    t = np.maximum(bs.years_from_days(out["dte"].clip(lower=0).values), 0.5 / 365)
    need_iv = ~np.isfinite(out.get("iv", pd.Series(np.nan, index=out.index)).astype(float))
    for i in np.where(need_iv.values)[0]:
        row = out.iloc[i]
        out.iat[i, out.columns.get_loc("iv")] = bs.implied_vol(
            float(row["mid"]), spot, float(row["strike"]), float(t[i]), r, q, row["type"])
    for greek in ("delta", "gamma", "theta", "vega"):
        if greek not in out.columns:
            out[greek] = np.nan
    for kind in ("call", "put"):
        m = (out["type"] == kind).values
        miss = m & ~np.isfinite(out["delta"].astype(float).values)
        if miss.any():
            gg = bs.greeks(spot, out.loc[miss, "strike"].values, t[miss],
                           out.loc[miss, "iv"].values, r, q, kind)
            for k in ("delta", "gamma", "theta", "vega"):
                out.loc[miss, k] = gg[k]
    out = out[np.isfinite(out["iv"]) & (out["iv"] > 0.01)].reset_index(drop=True)
    out.attrs.update(chain.attrs)
    out.attrs["source"] = "polygon"
    return out


def get_chain(spot, vix, dte, *, vix_baseline=None, r=0.04, q=0.013,
              min_dte=3, max_dte=9, prefer_live=True):
    """Live Polygon chain when a key is set and it works; else the VIX
    reconstruction. Both come back with the columns score_chain expects.
    """
    if prefer_live and _poly.available():
        try:
            raw = _poly.nearest_weekly_chain(min_dte=min_dte, max_dte=max_dte)
            return prep_polygon_chain(raw, r, q)
        except Exception:      # noqa: BLE001 -- any API hiccup -> fall back
            pass
    return synthetic_chain(spot, vix, dte, vix_baseline=vix_baseline, r=r, q=q)


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
