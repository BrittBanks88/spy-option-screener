"""Polygon.io adapter — real-time SPY quote, intraday bars, option chains + Greeks.

Set ``POLYGON_API_KEY`` in .env (or the environment). The rest of the package
auto-uses this whenever ``available()`` is true and falls back to yfinance
otherwise, so nothing else needs a code change to go live.

Plan-agnostic: the same endpoints return real-time data on a real-time options
plan and 15-minute-delayed data on lower tiers. ``quote_age_seconds()`` on a
chain tells you which you got.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache

import numpy as np
import pandas as pd

BASE = "https://api.polygon.io"
ET = "America/New_York"
RTH_OPEN, RTH_CLOSE = dt.time(9, 30), dt.time(16, 0)


def available() -> bool:
    return bool(os.environ.get("POLYGON_API_KEY"))


def _key() -> str:
    k = os.environ.get("POLYGON_API_KEY")
    if not k:
        raise RuntimeError("POLYGON_API_KEY not set")
    return k


def _get(path: str, _tries: int = 4, **params) -> dict:
    """GET a Polygon endpoint (path or absolute next_url). Retries 429/5xx."""
    if path.startswith("http"):
        url = path + ("&" if "?" in path else "?") + "apiKey=" + _key()
    else:
        params["apiKey"] = _key()
        url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(_tries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:   # noqa: S310
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < _tries - 1:
                time.sleep(2 ** attempt * 3)
                continue
            raise RuntimeError(f"Polygon {e.code} on {path}: {e.read()[:200]!r}")
    raise RuntimeError(f"Polygon: exhausted retries on {path}")


def _paginate(path: str, **params):
    data = _get(path, **params)
    yield from data.get("results", []) or []
    nxt = data.get("next_url")
    while nxt:
        data = _get(nxt)
        yield from data.get("results", []) or []
        nxt = data.get("next_url")


# ── underlying ─────────────────────────────────────────────────────────────
def spot() -> float:
    """Latest SPY trade price."""
    d = _get("/v2/last/trade/SPY")
    return float(d["results"]["p"])


def intraday_bars(interval: str = "30m", day: dt.date | None = None) -> pd.DataFrame:
    """1-day (or ``day``) intraday SPY bars, ET, regular hours only.
    Columns Open/High/Low/Close/Volume, plus ``date`` and ``minute``.
    """
    mult, span = {"1m": (1, "minute"), "5m": (5, "minute"), "15m": (15, "minute"),
                  "30m": (30, "minute"), "1h": (60, "minute"),
                  "1d": (1, "day")}[interval]
    day = day or dt.datetime.now().astimezone().date()
    rows = _get(f"/v2/aggs/ticker/SPY/range/{mult}/{span}/{day}/{day}",
                adjusted="true", sort="asc", limit=50000).get("results", []) or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={
        "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "t": "ts"})
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(ET)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df[(df.index.time >= RTH_OPEN) & (df.index.time < RTH_CLOSE)].copy()
    df["date"] = df.index.date
    df["minute"] = (df.index.hour - 9) * 60 + df.index.minute - 30
    return df


# ── options ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def upcoming_expiries(n: int = 8) -> tuple[dt.date, ...]:
    today = dt.date.today().isoformat()
    seen = []
    for c in _paginate("/v3/reference/options/contracts",
                       underlying_ticker="SPY", expired="false",
                       **{"expiration_date.gte": today}, limit=1000,
                       sort="expiration_date", order="asc"):
        e = dt.date.fromisoformat(c["expiration_date"])
        if e not in seen:
            seen.append(e)
        if len(seen) >= n:
            break
    return tuple(seen)


def option_chain(expiry: dt.date, strike_band: float = 0.08) -> pd.DataFrame:
    """Snapshot of the SPY chain for one expiry. Columns:
        expiry, dte, type, strike, bid, ask, mid, last, volume, open_interest,
        iv, delta, gamma, theta, vega, quote_ts
    ``.attrs``: spot, atm_vol, delayed (bool).
    """
    und = spot()
    lo, hi = und * (1 - strike_band), und * (1 + strike_band)
    rows = list(_paginate(
        "/v3/snapshot/options/SPY",
        expiration_date=expiry.isoformat(),
        **{"strike_price.gte": round(lo), "strike_price.lte": round(hi)},
        limit=250))
    if not rows:
        raise RuntimeError(f"Polygon: empty chain for {expiry}")

    today = dt.date.today()
    out = []
    newest_ns = 0
    for r in rows:
        det = r.get("details", {})
        q = r.get("last_quote", {}) or {}
        g = r.get("greeks", {}) or {}
        bid, ask = q.get("bid"), q.get("ask")
        mid = q.get("midpoint")
        if mid is None and bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
        last = (r.get("last_trade") or {}).get("price")
        newest_ns = max(newest_ns, q.get("last_updated", 0) or 0)
        out.append({
            "expiry": expiry, "dte": (expiry - today).days,
            "type": det.get("contract_type"),
            "strike": float(det.get("strike_price")),
            "bid": bid, "ask": ask,
            "mid": mid if mid is not None else last,
            "last": last,
            "volume": (r.get("day") or {}).get("volume"),
            "open_interest": r.get("open_interest"),
            "iv": r.get("implied_volatility"),
            "delta": g.get("delta"), "gamma": g.get("gamma"),
            "theta": g.get("theta"), "vega": g.get("vega"),
        })
    df = pd.DataFrame(out).dropna(subset=["strike", "type"])
    df = df[df["mid"].fillna(0) > 0].sort_values(["type", "strike"]).reset_index(drop=True)

    atm = df.loc[(df["strike"] - und).abs() < und * 0.01, "iv"].dropna()
    df.attrs["spot"] = float(und)
    df.attrs["atm_vol"] = float(atm.median()) if len(atm) else float(
        df["iv"].dropna().median() if df["iv"].notna().any() else 0.15)
    if newest_ns:
        age = time.time() - newest_ns / 1e9
        df.attrs["quote_age_seconds"] = age
        df.attrs["delayed"] = age > 120
    return df


def nearest_weekly_chain(min_dte: int = 3, max_dte: int = 9,
                         strike_band: float = 0.08) -> pd.DataFrame:
    today = dt.date.today()
    for e in upcoming_expiries():
        if min_dte <= (e - today).days <= max_dte:
            return option_chain(e, strike_band)
    return option_chain(upcoming_expiries()[0], strike_band)
