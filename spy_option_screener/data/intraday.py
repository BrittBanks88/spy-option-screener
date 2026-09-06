"""Intraday SPY bars + per-day intraday features (gap, opening range, timed price).

yfinance limits: 1h bars ~2 years, 30m/15m ~60 days, 1m ~7 days. We cache what
we pull. Everything is US/Eastern, regular trading hours only (09:30-16:00).

Feature frame (one row per session date) powers:
  * entry-time backtesting  (price at 10:00 instead of the close)
  * gap-continuation signal
  * opening-range-breakout signal
  * the "today's pick" surface
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
CACHE_DIR.mkdir(exist_ok=True)
ET = "America/New_York"
RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)

_PERIOD = {"1h": "720d", "30m": "60d", "15m": "60d", "5m": "60d", "1m": "7d"}


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def load_intraday(interval="1h", refresh=False, period=None,
                  live_today=False) -> pd.DataFrame:
    """Cached intraday SPY bars, ET, regular hours. Columns: Open/High/Low/
    Close/Volume plus ``date`` (session date) and ``minute`` (minutes since 9:30).

    ``live_today``: if a POLYGON_API_KEY is set, swap today's rows for real-time
    Polygon bars (yfinance intraday is ~15 min delayed).
    """
    period = period or _PERIOD.get(interval, "60d")
    cache = CACHE_DIR / f"SPY_{interval}.parquet"

    if refresh or not cache.exists():
        import yfinance as yf
        raw = yf.download("SPY", interval=interval, period=period,
                          progress=False, auto_adjust=False)
        raw = _flatten(raw)
        if raw.empty:
            raise RuntimeError(f"yfinance returned no {interval} SPY data")
        raw.index = (raw.index.tz_convert(ET) if raw.index.tz
                     else raw.index.tz_localize("UTC").tz_convert(ET))
        raw.to_parquet(cache)

    df = pd.read_parquet(cache)
    df = df[(df.index.time >= RTH_OPEN) & (df.index.time < RTH_CLOSE)].copy()
    df["date"] = df.index.date
    df["minute"] = (df.index.hour - 9) * 60 + df.index.minute - 30

    if live_today:
        df = _append_live_today(df, interval)
    return df


def _append_live_today(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Replace today's (delayed/absent) rows with real-time vendor bars."""
    from . import polygon, schwab
    vendor = schwab if schwab.available() else polygon if polygon.available() else None
    if vendor is None:
        return df
    try:
        fresh = vendor.intraday_bars(interval if interval in
                                     ("1m", "5m", "15m", "30m", "1h") else "30m")
    except Exception:      # noqa: BLE001
        return df
    if fresh.empty:
        return df
    today = fresh["date"].iloc[0]
    df = df[df["date"] != today]
    return pd.concat([df, fresh]).sort_index()


def _price_at_or_after(day: pd.DataFrame, hhmm: str) -> float:
    """Close of the first bar whose timestamp is >= hhmm (fallback: last bar)."""
    t = dt.time(*map(int, hhmm.split(":")))
    after = day[day.index.time >= t]
    row = after.iloc[0] if len(after) else day.iloc[-1]
    return float(row["Close"])


def daily_features(intraday: pd.DataFrame, daily: pd.DataFrame,
                   or_minutes: int = 30, signal_cutoff: str = "10:30",
                   entry_times=("10:00", "10:30", "11:00")) -> pd.DataFrame:
    """One row per session.

    SIGNAL-SAFE fields (known by ``signal_cutoff``, safe to trade on):
        prev_close, open, gap, or_high, or_low, or_width,
        orb_dir, orb_break_minute, ret_open_to_cutoff, px_HHMM
    OUTCOME fields (whole session, for labelling / research only -- never feed
    these to a signal): first_hour_ret, close, day_high, day_low, close_vs_or.

    ``daily`` supplies the prior close (gap) and the daily VIX / realized vol.
    """
    daily = daily.copy()
    daily.index = pd.to_datetime(daily.index).date
    cutoff_t = dt.time(*map(int, signal_cutoff.split(":")))
    cutoff_min = (cutoff_t.hour - 9) * 60 + cutoff_t.minute - 30
    rows = []
    for date, day in intraday.groupby("date"):
        day = day.sort_index()
        if len(day) < 2:
            continue
        op = float(day["Open"].iloc[0])
        cl = float(day["Close"].iloc[-1])
        orr = day[day["minute"] < or_minutes]
        if orr.empty:
            orr = day.iloc[:1]
        or_hi, or_lo = float(orr["High"].max()), float(orr["Low"].min())

        prev_close = float(daily["close"].get(_prev_session(daily, date), np.nan))
        vix = float(daily["vix"].get(date, np.nan))
        rv = float(daily["realized_vol_20"].get(date, np.nan))

        # opening-range break, considering ONLY bars up to the signal cutoff
        pre = day[(day["minute"] >= or_minutes) & (day["minute"] <= cutoff_min)]
        broke_up = pre[pre["Close"] > or_hi]
        broke_dn = pre[pre["Close"] < or_lo]
        orb_dir, first_break_min = 0, np.nan
        if len(broke_up) and (not len(broke_dn)
                              or broke_up.index[0] < broke_dn.index[0]):
            orb_dir, first_break_min = 1, int(broke_up.iloc[0]["minute"])
        elif len(broke_dn):
            orb_dir, first_break_min = -1, int(broke_dn.iloc[0]["minute"])

        rec = {
            "date": pd.Timestamp(date),
            "prev_close": prev_close,
            "open": op,
            "gap": op / prev_close - 1.0 if prev_close else np.nan,
            "or_high": or_hi,
            "or_low": or_lo,
            "or_width": (or_hi / or_lo - 1.0) if or_lo else np.nan,
            "orb_dir": orb_dir,
            "orb_break_minute": first_break_min,
            "ret_open_to_cutoff": _price_at_or_after(day, signal_cutoff) / op - 1.0,
            # outcome-only
            "first_hour_ret": _price_at_or_after(day, "10:30") / op - 1.0,
            "close_vs_or": (cl > or_hi) - (cl < or_lo),
            "day_high": float(day["High"].max()),
            "day_low": float(day["Low"].min()),
            "close": cl,
            "vix": vix,
            "realized_vol_20": rv,
        }
        for et in entry_times:
            rec[f"px_{et.replace(':', '')}"] = _price_at_or_after(day, et)
        rows.append(rec)

    return pd.DataFrame(rows).set_index("date").sort_index()


def _prev_session(daily: pd.DataFrame, date) -> object:
    idx = list(daily.index)
    try:
        i = idx.index(date)
        return idx[i - 1] if i > 0 else date
    except ValueError:
        prior = [d for d in idx if d < date]
        return prior[-1] if prior else date


def latest_session(interval="30m", refresh=False, live_today=False):
    """(date, day_bars) for the most recent session with data. Used by the
    'today's pick' surface when run outside market hours.
    """
    df = load_intraday(interval, refresh=refresh, live_today=live_today)
    last_date = df["date"].max()
    return last_date, df[df["date"] == last_date].sort_index()
