"""Offline tests for the Schwab adapter — token handling + chain parsing.
No network: `_get` and the token store are monkeypatched.
"""
import datetime as dt
import time

import pytest

from spy_option_screener.data import schwab


def test_available_false_without_creds(monkeypatch):
    monkeypatch.delenv("SCHWAB_APP_KEY", raising=False)
    monkeypatch.delenv("SCHWAB_APP_SECRET", raising=False)
    assert schwab.available() is False


def test_available_needs_unexpired_refresh_token(monkeypatch):
    monkeypatch.setenv("SCHWAB_APP_KEY", "k")
    monkeypatch.setenv("SCHWAB_APP_SECRET", "s")
    monkeypatch.setattr(schwab, "_load_token",
                        lambda: {"refresh_expires_at": time.time() - 10})
    assert schwab.available() is False
    monkeypatch.setattr(schwab, "_load_token",
                        lambda: {"refresh_expires_at": time.time() + 86400})
    assert schwab.available() is True


_CHAIN = {
    "underlyingPrice": 763.7,
    "callExpDateMap": {
        "2026-09-08:3": {
            "764.0": [{"strikePrice": 764.0, "bid": 3.6, "ask": 3.8, "mark": 3.7,
                       "last": 3.7, "totalVolume": 5000, "openInterest": 12000,
                       "volatility": 13.7, "delta": 0.51, "gamma": 0.04,
                       "theta": -0.66, "vega": 0.28}],
            "999.0": [{"strikePrice": 999.0, "bid": 0, "ask": 0, "mark": 0,
                       "volatility": -999, "delta": -999}],
        }},
    "putExpDateMap": {
        "2026-09-08:3": {
            "764.0": [{"strikePrice": 764.0, "bid": 3.1, "ask": 3.3, "mark": 3.2,
                       "last": 3.2, "totalVolume": 3000, "openInterest": 8000,
                       "volatility": 13.9, "delta": -0.49, "gamma": 0.04,
                       "theta": -0.60, "vega": 0.28}]}},
}


def test_option_chain_parsing(monkeypatch):
    monkeypatch.setenv("SCHWAB_APP_KEY", "k")
    monkeypatch.setenv("SCHWAB_APP_SECRET", "s")
    monkeypatch.setattr(schwab, "_get", lambda p, _tries=3, **kw: _CHAIN)

    df = schwab.option_chain(dt.date(2026, 9, 8))
    assert set(df["type"]) == {"call", "put"}          # junk strike dropped
    call = df[(df["type"] == "call") & (df["strike"] == 764)].iloc[0]
    assert call["mid"] == pytest.approx(3.7)
    assert call["iv"] == pytest.approx(0.137)          # % -> decimal
    assert call["delta"] == pytest.approx(0.51)
    assert df.attrs["spot"] == pytest.approx(763.7)
    assert df.attrs["delayed"] is False
    assert df.attrs["quote_age_seconds"] == 0.0


def test_prep_live_chain_schwab_source(monkeypatch):
    monkeypatch.setenv("SCHWAB_APP_KEY", "k")
    monkeypatch.setenv("SCHWAB_APP_SECRET", "s")
    monkeypatch.setattr(schwab, "_get", lambda p, _tries=3, **kw: _CHAIN)
    from spy_option_screener.screener import chain as chain_mod, score as score_mod
    raw = schwab.option_chain(dt.date(2026, 9, 8))
    prepped = chain_mod.prep_live_chain(raw, source="schwab")
    assert prepped.attrs["source"] == "schwab"
    assert prepped.attrs["reanchored"] is False        # real-time -> no re-anchor
    sc = score_mod.score_chain(prepped, 1, 763.7, 0.11)
    assert "score" in sc.columns


def test_get_chain_prefers_schwab_over_polygon(monkeypatch):
    from spy_option_screener.data import polygon
    from spy_option_screener.screener import chain as chain_mod
    monkeypatch.setattr(schwab, "available", lambda: True)
    monkeypatch.setattr(polygon, "available", lambda: True)
    monkeypatch.setattr(schwab, "nearest_weekly_chain",
                        lambda **kw: schwab.option_chain(dt.date(2026, 9, 8)))
    monkeypatch.setattr(schwab, "_get", lambda p, _tries=3, **kw: _CHAIN)
    monkeypatch.setattr("spy_option_screener.data.loader.live_spot", lambda: None)
    ch = chain_mod.get_chain(763.0, 15.0, 3)
    assert ch.attrs["source"] == "schwab"
