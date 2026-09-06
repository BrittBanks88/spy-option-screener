"""Charles Schwab Trader API adapter — free real-time SPY quote, intraday bars,
and option chains with Greeks (for account holders).

Setup (one time): register an app at developer.schwab.com, put the keys in .env

    SCHWAB_APP_KEY=...
    SCHWAB_APP_SECRET=...
    SCHWAB_CALLBACK_URL=https://127.0.0.1        # must match the app registration

then run ``python schwab_auth.py`` and authorise in the browser. That stores a
token in cache/schwab_token.json. The access token auto-refreshes; the refresh
token lasts 7 days, so re-run ``schwab_auth.py`` about weekly.

``available()`` is False (and everything falls back to Polygon / the VIX
reconstruction) whenever the token is missing or its refresh token has expired.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "https://api.schwabapi.com"
AUTH_URL = f"{ROOT}/v1/oauth/authorize"
TOKEN_URL = f"{ROOT}/v1/oauth/token"
MKT = f"{ROOT}/marketdata/v1"
ET = "America/New_York"
RTH_OPEN, RTH_CLOSE = dt.time(9, 30), dt.time(16, 0)

TOKEN_PATH = Path(__file__).resolve().parents[2] / "cache" / "schwab_token.json"


# ── credentials / token store ─────────────────────────────────────────────
def _creds() -> tuple[str, str, str]:
    k, s = os.environ.get("SCHWAB_APP_KEY"), os.environ.get("SCHWAB_APP_SECRET")
    cb = os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1")
    if not (k and s):
        raise RuntimeError("SCHWAB_APP_KEY / SCHWAB_APP_SECRET not set")
    return k, s, cb


def _basic_auth() -> str:
    k, s, _ = _creds()
    return base64.b64encode(f"{k}:{s}".encode()).decode()


def _load_token() -> dict | None:
    try:
        return json.loads(TOKEN_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _save_token(tok: dict) -> None:
    TOKEN_PATH.parent.mkdir(exist_ok=True)
    TOKEN_PATH.write_text(json.dumps(tok, indent=2))


def _token_request(data: dict) -> dict:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(TOKEN_URL, data=body, headers={
        "Authorization": f"Basic {_basic_auth()}",
        "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=20) as r:      # noqa: S310
        return json.loads(r.read())


def _store_from_response(resp: dict, keep_refresh_clock: dict | None = None) -> dict:
    now = time.time()
    tok = {
        "access_token": resp["access_token"],
        "refresh_token": resp.get("refresh_token")
        or (keep_refresh_clock or {}).get("refresh_token"),
        "access_expires_at": now + int(resp.get("expires_in", 1800)) - 60,
    }
    if resp.get("refresh_token") or not keep_refresh_clock:
        tok["refresh_expires_at"] = now + 7 * 86400 - 300
    else:
        tok["refresh_expires_at"] = keep_refresh_clock.get(
            "refresh_expires_at", now + 7 * 86400 - 300)
    _save_token(tok)
    return tok


# ── public auth helpers (used by schwab_auth.py) ─────────────────────────
def authorize_url() -> str:
    k, _, cb = _creds()
    return f"{AUTH_URL}?{urllib.parse.urlencode({'client_id': k, 'redirect_uri': cb})}"


def exchange_code(redirect_response: str) -> dict:
    """Take the full URL Schwab redirected to (contains ?code=...) -> save token."""
    q = urllib.parse.urlparse(redirect_response.strip()).query
    code = urllib.parse.parse_qs(q).get("code", [None])[0]
    if not code:
        raise RuntimeError("no ?code= in that URL")
    _, _, cb = _creds()
    resp = _token_request({"grant_type": "authorization_code", "code": code,
                           "redirect_uri": cb})
    return _store_from_response(resp)


def refresh() -> dict:
    tok = _load_token()
    if not tok or not tok.get("refresh_token"):
        raise RuntimeError("no refresh token — run: python schwab_auth.py")
    if time.time() > tok.get("refresh_expires_at", 0):
        raise RuntimeError("refresh token expired — run: python schwab_auth.py")
    resp = _token_request({"grant_type": "refresh_token",
                           "refresh_token": tok["refresh_token"]})
    return _store_from_response(resp, keep_refresh_clock=tok)


def token_status() -> dict:
    tok = _load_token() or {}
    now = time.time()
    return {
        "have_token": bool(tok),
        "access_valid_for": max(0, tok.get("access_expires_at", 0) - now),
        "refresh_valid_for_days": max(0, (tok.get("refresh_expires_at", 0) - now) / 86400),
    }


# ── authed requests ─────────────────────────────────────────────────────
def available() -> bool:
    try:
        _creds()
    except RuntimeError:
        return False
    tok = _load_token()
    return bool(tok and time.time() < tok.get("refresh_expires_at", 0))


def _access_token() -> str:
    tok = _load_token()
    if not tok:
        raise RuntimeError("not authenticated — run: python schwab_auth.py")
    if time.time() >= tok.get("access_expires_at", 0):
        tok = refresh()
    return tok["access_token"]


def _get(path: str, _tries: int = 3, **params) -> dict:
    for attempt in range(_tries):
        url = f"{MKT}{path}?{urllib.parse.urlencode(params, doseq=True)}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {_access_token()}", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:   # noqa: S310
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 401 and attempt == 0:
                refresh()
                continue
            if e.code in (429, 500, 502, 503) and attempt < _tries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"Schwab {e.code} on {path}: {e.read()[:200]!r}")
    raise RuntimeError(f"Schwab: exhausted retries on {path}")


# ── underlying / bars ───────────────────────────────────────────────────
def spot() -> float:
    d = _get("/SPY/quotes")
    q = d["SPY"]["quote"]
    return float(q.get("lastPrice") or q.get("mark") or
                 (q["bidPrice"] + q["askPrice"]) / 2)


def intraday_bars(interval: str = "30m", day: dt.date | None = None) -> pd.DataFrame:
    freq = {"1m": 1, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 30}.get(interval, 30)
    day = day or dt.datetime.now().astimezone().date()
    start = int(dt.datetime.combine(day, dt.time(0)).timestamp() * 1000)
    end = int(dt.datetime.combine(day, dt.time(23, 59)).timestamp() * 1000)
    d = _get("/pricehistory", symbol="SPY", periodType="day", frequencyType="minute",
             frequency=freq, needExtendedHoursData="false",
             startDate=start, endDate=end)
    candles = d.get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles).rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "datetime": "ts"})
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(ET)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df[(df.index.time >= RTH_OPEN) & (df.index.time < RTH_CLOSE)].copy()
    df["date"] = df.index.date
    df["minute"] = (df.index.hour - 9) * 60 + df.index.minute - 30
    return df


# ── options ────────────────────────────────────────────────────────────
def _flatten_exp_map(exp_map: dict, kind: str, today: dt.date) -> list[dict]:
    rows = []
    for exp_key, strikes in exp_map.items():
        exp = dt.date.fromisoformat(exp_key.split(":")[0])
        for _strike, contracts in strikes.items():
            c = contracts[0]
            iv = c.get("volatility")
            iv = None if iv in (None, -999, -999.0) else iv / 100.0
            bid, ask = c.get("bid"), c.get("ask")
            mid = c.get("mark")
            if mid in (None, 0) and bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
            rows.append({
                "expiry": exp, "dte": (exp - today).days, "type": kind,
                "strike": float(c["strikePrice"]),
                "bid": bid, "ask": ask, "mid": mid, "last": c.get("last"),
                "volume": c.get("totalVolume"), "open_interest": c.get("openInterest"),
                "iv": iv,
                "delta": _g(c, "delta"), "gamma": _g(c, "gamma"),
                "theta": _g(c, "theta"), "vega": _g(c, "vega"),
            })
    return rows


def _g(c: dict, k: str):
    v = c.get(k)
    return None if v in (None, -999, -999.0) else v


def option_chain(expiry: dt.date, strike_band: float = 0.08) -> pd.DataFrame:
    d = _get("/chains", symbol="SPY", contractType="ALL",
             includeUnderlyingQuote="true",
             fromDate=expiry.isoformat(), toDate=expiry.isoformat())
    und = float(d.get("underlyingPrice")
                or d.get("underlying", {}).get("last") or spot())
    today = dt.date.today()
    rows = (_flatten_exp_map(d.get("callExpDateMap", {}), "call", today)
            + _flatten_exp_map(d.get("putExpDateMap", {}), "put", today))
    df = pd.DataFrame(rows).dropna(subset=["strike", "type"])
    lo, hi = und * (1 - strike_band), und * (1 + strike_band)
    df = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
    df = df[df["mid"].fillna(0) > 0].sort_values(["type", "strike"]).reset_index(drop=True)

    atm = df.loc[(df["strike"] - und).abs() < und * 0.01, "iv"].dropna()
    df.attrs["spot"] = und
    df.attrs["atm_vol"] = float(atm.median()) if len(atm) else float(
        df["iv"].dropna().median() if df["iv"].notna().any() else 0.15)
    df.attrs["quote_age_seconds"] = 0.0        # Schwab market data is real-time
    df.attrs["delayed"] = False
    return df


def nearest_weekly_chain(min_dte: int = 3, max_dte: int = 9,
                         strike_band: float = 0.08) -> pd.DataFrame:
    today = dt.date.today()
    for e in upcoming_expiries():
        if min_dte <= (e - today).days <= max_dte:
            return option_chain(e, strike_band)
    return option_chain(upcoming_expiries()[0], strike_band)


def upcoming_expiries(n: int = 8) -> list[dt.date]:
    today = dt.date.today()
    d = _get("/chains", symbol="SPY", contractType="CALL", strikeCount=1,
             fromDate=today.isoformat(),
             toDate=(today + dt.timedelta(days=45)).isoformat())
    keys = sorted(set(d.get("callExpDateMap", {}).keys()))
    return [dt.date.fromisoformat(k.split(":")[0]) for k in keys][:n]
