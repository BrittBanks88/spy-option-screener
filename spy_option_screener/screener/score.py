"""Rank long call / put contracts for a given directional view.

The philosophy: a long option is a good buy when the move you expect is large
relative to what the option costs you to hold. Concretely we reward

  * breakeven within the expected move (you don't need a miracle to profit)
  * enough delta to actually capture the move (leverage without lottery odds)
  * low theta bleed as a fraction of premium (time is not crushing you fast)
  * IV that is cheap vs SPY's own recent realized vol (not overpaying for vol)
  * tight, liquid markets (you can get in and out)

and penalize the opposite. Output is a 0..100 score plus every component so
you can see WHY a contract ranks where it does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..pricing import vol_model as vm


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def score_chain(chain: pd.DataFrame, direction: int, spot: float,
                realized_vol: float, horizon_days: int | None = None,
                confidence: float = 1.0, weights: dict | None = None,
                r: float = 0.04, q: float = 0.013) -> pd.DataFrame:
    """Score the side of the chain matching ``direction`` (+1 call, -1 put).

    ``realized_vol`` is SPY's recent annualized realized vol -- the yardstick
    for whether option IV is cheap or rich. ``horizon_days`` defaults to the
    contract DTE (assume you hold to expiry-ish).
    """
    w = {
        "breakeven": 0.30,
        "delta": 0.15,
        "theta": 0.20,
        "vol_value": 0.20,
        "liquidity": 0.15,
    }
    if weights:
        w.update(weights)

    kind = "call" if direction > 0 else "put"
    c = chain[chain["type"] == kind].copy()
    if c.empty:
        return c

    atm_vol = float(chain.attrs.get("atm_vol", c["iv"].median()))
    dte = c["dte"].astype(float)
    # Horizon for the "expected move" yardstick: hold-to-expiry unless the
    # caller specifies a shorter planned holding period.
    hz = dte.values if not horizon_days else np.minimum(dte.values, float(horizon_days))

    exp_move = vm.expected_move(spot, atm_vol, hz)  # 1-sigma $ move over horizon

    # Breakeven distance relative to expected move -----------------------------
    prem = c["mid"].astype(float)
    if kind == "call":
        breakeven = c["strike"] + prem
        be_move = breakeven - spot
    else:
        breakeven = c["strike"] - prem
        be_move = spot - breakeven
    # be_ratio < 1 -> breakeven is inside a 1-sigma move (good)
    exp_move = np.where(np.isfinite(exp_move) & (exp_move > 1e-6), exp_move, np.nan)
    be_ratio = (be_move / exp_move).clip(-5, 20)
    be_score = _clip01(1.4 - be_ratio.fillna(3.0))  # missing move -> harsh

    # Delta -- want meaningful participation, not a 0.05-delta lotto ----------
    adelta = c["delta"].abs()
    delta_score = _clip01(1.0 - ((adelta - 0.45).abs() / 0.45))

    # Theta bleed as fraction of premium per day -----------------------------
    theta_frac = c["theta"].abs() / np.maximum(prem, 1e-6)
    theta_score = _clip01(1.0 - theta_frac / 0.12)  # 0.12/day premium loss -> 0

    # Vol value: IV vs recent realized vol -----------------------------------
    iv_vs_rv = c["iv"] / max(realized_vol, 1e-6)
    vol_score = _clip01(1.3 - iv_vs_rv)  # iv==rv -> 0.3 ; iv 30% under rv -> ~0.6

    # Liquidity -------------------------------------------------------------
    if "spread_pct" in c:
        liq = _clip01(1.0 - c["spread_pct"].fillna(1.0) / 0.15)
        if "open_interest" in c:
            oi = _clip01(np.log1p(c["open_interest"].fillna(0)) / np.log1p(20000))
            liq = 0.6 * liq + 0.4 * oi
    else:
        # synthetic chain: proxy liquidity by closeness to the money
        liq = _clip01(1.0 - (c["moneyness"] - 1.0).abs() / 0.06)

    raw = (
        w["breakeven"] * be_score
        + w["delta"] * delta_score
        + w["theta"] * theta_score
        + w["vol_value"] * vol_score
        + w["liquidity"] * liq
    ) / sum(w.values())

    # Robinhood-style trade math -------------------------------------------
    T = np.maximum(dte.values / 365.0, 1e-6)
    ivc = c["iv"].astype(float).values
    # P(finishing past breakeven) under a lognormal terminal distribution.
    d2 = (np.log(spot / breakeven.values) + (r - q - 0.5 * ivc**2) * T) / (ivc * np.sqrt(T))
    pop = norm.cdf(d2) if kind == "call" else norm.cdf(-d2)

    c["exp_move"] = exp_move
    c["breakeven"] = breakeven
    c["be_move_needed"] = be_move
    c["be_pct"] = be_move / spot                     # % move to breakeven
    c["be_ratio"] = be_ratio
    c["chance_of_profit"] = pop
    c["max_loss"] = prem * 100.0                     # per contract, long
    c["cost"] = prem * 100.0
    c["leverage"] = (c["delta"].abs() * spot) / np.maximum(prem, 1e-6)
    c["s_breakeven"] = be_score
    c["s_delta"] = delta_score
    c["s_theta"] = theta_score
    c["s_vol_value"] = vol_score
    c["s_liquidity"] = liq
    c["score"] = (100.0 * raw * _clip01(0.3 + 0.7 * confidence)).round(1)
    return c.sort_values("score", ascending=False).reset_index(drop=True)


def top_pick(scored: pd.DataFrame) -> pd.Series | None:
    return None if scored is None or scored.empty else scored.iloc[0]
