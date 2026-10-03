"""Development-only SPY option chain, plus the one chain schema everything
downstream (screener/contract_picker.py) expects.

THE SCHEMA -- a real vendor adapter (Massive/Polygon, Schwab, ...) only has to
return a DataFrame with these columns, one row per contract:

    type            "call" | "put"
    expiry          expiration date (date or Timestamp)
    dte             calendar days from today to expiry
    strike          float
    bid, ask        float, per share (x100 for one contract)
    iv              implied volatility as a decimal (0.14 = 14%)
    -- optional but used when present --
    mid             (bid+ask)/2, computed if missing
    delta           signed (puts negative); computed from iv if missing
    open_interest   int
    volume          int

plus ``chain.attrs``: ``spot`` (float), ``source`` (str), and optionally
``quote_age_seconds`` (float) so stale quotes can be flagged.

The numbers here are MODEL OUTPUT dressed up with plausible spreads and
liquidity so the picker can be built and tested with no data subscription.
They are not market quotes. ``attrs["source"] == "fake"`` makes the picker
warn loudly about that.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..screener.chain import synthetic_chain

REQUIRED_COLUMNS = ("type", "expiry", "dte", "strike", "bid", "ask", "iv")
DEFAULT_DTES = (3, 7, 10, 14, 17, 21, 28, 35)


def fake_chain(spot: float, *, today: dt.date | None = None, vix: float = 15.0,
               dtes=DEFAULT_DTES) -> pd.DataFrame:
    """Deterministic fake SPY chain around ``spot`` for the given expiries."""
    today = today or dt.date.today()
    parts = []
    for dte in dtes:
        base = synthetic_chain(spot, vix, dte, width=0.06, step=1.0)
        base = base[base["mid"] > 0.02].copy()
        spread = 0.02 + 0.015 * base["mid"]                  # wider when pricier
        base["bid"] = (base["mid"] - spread / 2).clip(lower=0.01)
        base["ask"] = base["mid"] + spread / 2
        base["mid"] = (base["bid"] + base["ask"]) / 2
        near = np.exp(-(((base["moneyness"] - 1.0) / 0.03) ** 2))   # liquid near the money
        base["open_interest"] = (6000 * near * (1 + dte / 30)).round().astype(int)
        base["volume"] = (base["open_interest"] * 0.3).round().astype(int)
        base["expiry"] = pd.Timestamp(today) + pd.Timedelta(days=int(dte))
        parts.append(base)
    chain = pd.concat(parts, ignore_index=True)
    chain = chain[["type", "expiry", "dte", "strike", "bid", "ask", "mid", "iv",
                   "delta", "gamma", "theta", "vega", "open_interest", "volume"]]
    chain.attrs.update(spot=float(spot), source="fake", quote_age_seconds=0.0)
    return chain
