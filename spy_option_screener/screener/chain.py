"""Build an option chain -- synthetic (for backtests) or real (for live use)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..pricing import black_scholes as bs
from ..pricing import vol_model as vm
from ..data import polygon as _poly
from ..data import schwab as _schwab


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


def _reanchor_to_live_spot(out: pd.DataFrame, snap_spot: float, live_spot: float,
                           r: float, q: float) -> pd.DataFrame:
    """Re-price a (possibly delayed) chain to a fresher underlying print.

    Keeps each strike's implied vol but slides the whole smile with spot
    (sticky-delta, the right assumption for index options intraday), then
    recomputes mid + Greeks at the live spot. bid/ask keep the real quoted
    spread % around the new mid.
    """
    dS = live_spot - snap_spot
    t = np.maximum(bs.years_from_days(out["dte"].clip(lower=0).values), 0.5 / 365)

    # sticky-delta: IV at strike K under the new spot ~ old IV at (K - dS)
    ks = out["strike"].to_numpy(float)
    order = np.argsort(ks)
    iv_new = np.interp(ks - dS, ks[order], out["iv"].to_numpy(float)[order])
    out = out.copy()
    out["iv"] = np.clip(iv_new, 0.01, 3.0)

    for kind in ("call", "put"):
        m = (out["type"] == kind).to_numpy()
        if not m.any():
            continue
        px = bs.price(live_spot, ks[m], t[m], out.loc[m, "iv"].to_numpy(), r, q, kind)
        g = bs.greeks(live_spot, ks[m], t[m], out.loc[m, "iv"].to_numpy(), r, q, kind)
        out.loc[m, "mid"] = px
        for k in ("delta", "gamma", "theta", "vega"):
            out.loc[m, k] = g[k]

    sp = out.get("spread_pct", pd.Series(0.03, index=out.index)).fillna(0.03).clip(0, 1)
    out["bid"] = (out["mid"] * (1 - sp / 2)).clip(lower=0)
    out["ask"] = out["mid"] * (1 + sp / 2)
    out["moneyness"] = out["strike"] / live_spot
    out.attrs["snapshot_spot"] = float(snap_spot)
    out.attrs["spot"] = float(live_spot)
    out.attrs["reanchored"] = True
    return out


def prep_live_chain(chain: pd.DataFrame, *, source: str, r=0.04, q=0.013,
                    live_spot: float | None = None) -> pd.DataFrame:
    """A broker/vendor snapshot (Schwab, Polygon) already carries IV + Greeks.
    Fill any gaps with the model, add mid/spread/moneyness. If ``live_spot`` is
    a fresher underlying print and the snapshot looks stale, re-anchor to it.
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
    out.attrs["source"] = source
    out.attrs["reanchored"] = False

    age = chain.attrs.get("quote_age_seconds")
    stale = (age is not None) and (age > 90)      # only Polygon reports an age
    if (live_spot and np.isfinite(live_spot) and live_spot > 0
            and (stale or abs(live_spot - spot) / spot > 3e-4)):
        out = _reanchor_to_live_spot(out, spot, float(live_spot), r, q)
    return out


def prep_polygon_chain(chain, r=0.04, q=0.013, live_spot=None):
    """Back-compat shim."""
    return prep_live_chain(chain, source="polygon", r=r, q=q, live_spot=live_spot)


# vendors in priority order: real-time & free first
_LIVE_VENDORS = (("schwab", _schwab), ("polygon", _poly))


def get_chain(spot, vix, dte, *, vix_baseline=None, r=0.04, q=0.013,
              min_dte=3, max_dte=9, prefer_live=True):
    """First working real-time vendor (Schwab, then Polygon); else the VIX
    reconstruction. Everything comes back with the columns score_chain expects,
    and a stale vendor chain is re-anchored to a near-real-time SPY quote.
    """
    if prefer_live:
        from ..data import loader as _loader
        for name, vendor in _LIVE_VENDORS:
            if not vendor.available():
                continue
            try:
                raw = vendor.nearest_weekly_chain(min_dte=min_dte, max_dte=max_dte)
                return prep_live_chain(raw, source=name, r=r, q=q,
                                       live_spot=_loader.live_spot())
            except Exception:      # noqa: BLE001 -- any API hiccup -> next vendor
                continue
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
