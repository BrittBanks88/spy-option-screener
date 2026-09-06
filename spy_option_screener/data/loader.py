"""Historical + current market data for SPY and VIX.

Free tier: yfinance daily bars for SPY and ^VIX, cached to parquet so repeat
runs are offline and fast. Swap this module for a paid options-data client
(ThetaData / Polygon) later -- nothing else in the package imports yfinance.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_TICKERS = {"SPY": "SPY", "VIX": "^VIX"}


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}_daily.parquet"


def load_history(start="2005-01-01", end=None, refresh=False) -> pd.DataFrame:
    """Return a daily DataFrame indexed by date with columns:

        open, high, low, close, volume, vix, ret, realized_vol_20

    ``realized_vol_20`` is the annualized 20-trading-day close-to-close vol.
    Data is cached; pass refresh=True to re-download.
    """
    end = end or dt.date.today().isoformat()
    spy_p, vix_p = _cache_path("SPY"), _cache_path("VIX")

    if refresh or not spy_p.exists() or not vix_p.exists():
        import yfinance as yf

        spy = yf.download("SPY", start=start, end=end, auto_adjust=False,
                          progress=False)
        vix = yf.download("^VIX", start=start, end=end, auto_adjust=False,
                          progress=False)
        if spy.empty:
            raise RuntimeError("yfinance returned no SPY data (offline / rate limited?)")
        _flatten_columns(spy).to_parquet(spy_p)
        _flatten_columns(vix).to_parquet(vix_p)

    spy = pd.read_parquet(spy_p)
    vix = pd.read_parquet(vix_p)

    df = pd.DataFrame(index=spy.index)
    df["open"] = spy["Open"]
    df["high"] = spy["High"]
    df["low"] = spy["Low"]
    df["close"] = spy["Close"]
    df["volume"] = spy["Volume"]
    df["vix"] = vix["Close"].reindex(df.index).ffill()

    df["ret"] = df["close"].pct_change()
    df["realized_vol_20"] = (
        df["ret"].rolling(20).std() * np.sqrt(252.0)
    )
    df = df.loc[str(start):str(end)]
    return df.dropna(subset=["close"])


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns a MultiIndex (field, ticker); flatten it."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def live_spot():
    """Near-real-time SPY last price for re-anchoring a delayed vendor chain.
    Prefers a real-time broker quote (Schwab), then Polygon, then yfinance's
    underlying quote (which lags only seconds, unlike its option chain).
    Returns None if nothing works.
    """
    from . import schwab, polygon
    for vendor in (schwab, polygon):
        try:
            if vendor.available():
                return float(vendor.spot())
        except Exception:      # noqa: BLE001
            pass
    try:
        import yfinance as yf
        fi = yf.Ticker("SPY").fast_info
        px = fi.get("last_price") or fi.get("lastPrice")
        return float(px) if px else None
    except Exception:      # noqa: BLE001
        return None


def load_live_chain(dte_max=7):
    """Current SPY option chain from yfinance for expiries within ``dte_max`` days.

    Returns a DataFrame with columns: expiry, dte, type, strike, bid, ask, mid,
    last, volume, open_interest, iv (exchange-provided), plus the current spot.

    Note: yfinance quotes are delayed ~15 min and mid can be stale outside RTH.
    """
    import yfinance as yf

    tk = yf.Ticker("SPY")
    spot = float(tk.fast_info["last_price"])
    today = dt.date.today()
    rows = []
    for exp in tk.options:
        exp_d = dt.date.fromisoformat(exp)
        dte = (exp_d - today).days
        if dte < 0 or dte > dte_max:
            continue
        oc = tk.option_chain(exp)
        for kind, tbl in (("call", oc.calls), ("put", oc.puts)):
            t = tbl.copy()
            t["expiry"] = exp
            t["dte"] = dte
            t["type"] = kind
            rows.append(t)
    if not rows:
        raise RuntimeError(f"No SPY expiries within {dte_max} days")

    chain = pd.concat(rows, ignore_index=True)
    chain = chain.rename(columns={
        "lastPrice": "last", "openInterest": "open_interest",
        "impliedVolatility": "iv", "strike": "strike",
        "bid": "bid", "ask": "ask", "volume": "volume",
    })
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2.0
    chain.loc[chain["mid"] <= 0, "mid"] = chain["last"]
    cols = ["expiry", "dte", "type", "strike", "bid", "ask", "mid", "last",
            "volume", "open_interest", "iv"]
    chain = chain[cols].sort_values(["expiry", "type", "strike"]).reset_index(drop=True)
    chain.attrs["spot"] = spot
    return chain
