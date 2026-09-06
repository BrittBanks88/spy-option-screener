"""Offline tests for the Polygon adapter — response parsing + graceful fallback.
No network: `_get` is monkeypatched with canned Polygon payloads.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from spy_option_screener.data import polygon as poly


def test_available_reflects_env(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    assert poly.available() is False
    monkeypatch.setenv("POLYGON_API_KEY", "x")
    assert poly.available() is True


_SNAPSHOT = {
    "results": [
        {"details": {"contract_type": "call", "strike_price": 764,
                     "expiration_date": "2026-09-08",
                     "ticker": "O:SPY260908C00764000"},
         "last_quote": {"bid": 3.65, "ask": 3.75, "midpoint": 3.70,
                        "last_updated": 1_900_000_000_000_000_000},
         "last_trade": {"price": 3.70},
         "greeks": {"delta": 0.51, "gamma": 0.04, "theta": -0.66, "vega": 0.28},
         "implied_volatility": 0.137, "open_interest": 12345,
         "day": {"volume": 5000}},
        {"details": {"contract_type": "put", "strike_price": 764,
                     "expiration_date": "2026-09-08",
                     "ticker": "O:SPY260908P00764000"},
         "last_quote": {"bid": 3.10, "ask": 3.20, "midpoint": 3.15,
                        "last_updated": 1_900_000_000_000_000_000},
         "greeks": {"delta": -0.49, "gamma": 0.04, "theta": -0.60, "vega": 0.28},
         "implied_volatility": 0.139, "open_interest": 8000,
         "day": {"volume": 3000}},
        {"details": {"contract_type": "call", "strike_price": 999,
                     "expiration_date": "2026-09-08", "ticker": "junk"},
         "last_quote": {}, "greeks": {}, "implied_volatility": None},
    ]
}


def test_option_chain_parsing(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "x")
    monkeypatch.setattr(poly, "spot", lambda: 763.7)

    def fake_get(path, _tries=4, **params):
        if path.startswith("/v3/snapshot/options"):
            return _SNAPSHOT
        raise AssertionError(path)
    monkeypatch.setattr(poly, "_get", fake_get)

    df = poly.option_chain(dt.date(2026, 9, 8))
    assert set(df["type"]) == {"call", "put"}          # the junk row dropped
    assert (df["mid"] > 0).all()
    row = df[(df["type"] == "call") & (df["strike"] == 764)].iloc[0]
    assert row["mid"] == pytest.approx(3.70)
    assert row["delta"] == pytest.approx(0.51)
    assert 0.13 < df.attrs["atm_vol"] < 0.15
    assert df.attrs["spot"] == pytest.approx(763.7)


def test_prep_polygon_chain_keeps_real_greeks(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "x")
    monkeypatch.setattr(poly, "spot", lambda: 763.7)
    monkeypatch.setattr(poly, "_get", lambda p, _tries=4, **k: _SNAPSHOT)
    from spy_option_screener.screener import chain as chain_mod
    raw = poly.option_chain(dt.date(2026, 9, 8))
    prepped = chain_mod.prep_polygon_chain(raw)
    for col in ("delta", "gamma", "theta", "vega", "iv", "moneyness", "spread_pct"):
        assert col in prepped.columns
    assert prepped.attrs["source"] == "polygon"
    # score_chain runs on it
    from spy_option_screener.screener import score as score_mod
    sc = score_mod.score_chain(prepped, 1, 763.7, 0.11)
    assert "score" in sc.columns and (sc["score"] >= 0).all()


def test_get_chain_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    from spy_option_screener.screener import chain as chain_mod
    ch = chain_mod.get_chain(770.0, 15.0, 4, vix_baseline=16.0)
    assert ch.attrs["source"] == "reconstructed"
    assert len(ch) > 0
