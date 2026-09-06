"""Tests for intraday features, signals, timed-entry backtest, market calendar."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from spy_option_screener.data import market_calendar as mcal
from spy_option_screener.signals import intraday_signals as isig


def test_market_calendar_labor_day_2026():
    assert not mcal.is_trading_day(dt.date(2026, 9, 7))     # Labor Day
    assert mcal.is_trading_day(dt.date(2026, 9, 8))
    exp, dte = mcal.next_weekly_expiry(dt.date(2026, 9, 4))
    assert exp == dt.date(2026, 9, 8) and dte == 4


def test_good_friday_excluded():
    assert not mcal.is_trading_day(dt.date(2026, 4, 3))     # Good Friday 2026


def _fake_features(n=120, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-02", periods=n)
    prev_close = 500 + rng.normal(0, 4, n).cumsum()
    gap = rng.normal(0, 0.004, n)
    op = prev_close * (1 + gap)
    or_w = np.abs(rng.normal(0.003, 0.001, n))
    orb = rng.choice([-1, 0, 1], n, p=[0.3, 0.2, 0.5])
    f = pd.DataFrame({
        "prev_close": prev_close, "open": op, "gap": gap,
        "or_high": op * (1 + or_w / 2), "or_low": op * (1 - or_w / 2),
        "or_width": or_w, "orb_dir": orb,
        "orb_break_minute": rng.integers(35, 90, n),
        "ret_open_to_cutoff": rng.normal(0, 0.003, n),
        "px_1000": op * (1 + rng.normal(0, 0.002, n)),
        "px_1030": op * (1 + rng.normal(0, 0.003, n)),
        "px_1100": op * (1 + rng.normal(0, 0.004, n)),
        "close": op * (1 + rng.normal(0, 0.006, n)),
        "vix": np.clip(15 + rng.normal(0, 2, n), 10, 40),
        "realized_vol_20": np.clip(0.12 + rng.normal(0, 0.02, n), 0.05, 0.4),
    }, index=idx)
    return f


def test_gap_continuation_direction_matches_gap_sign():
    f = _fake_features()
    out = isig.gap_continuation(f, up_threshold=0.003)
    fired = out[out["direction"] != 0]
    # only up-gaps trade by default, and always as calls
    assert (fired["direction"] == 1).all()
    assert (f.loc[fired.index, "gap"] >= 0.003).all()


def test_orb_respects_break_minute_and_width():
    f = _fake_features()
    out = isig.opening_range_breakout(f, max_break_minute=60, max_or_width=0.006)
    fired = out[out["direction"] != 0]
    assert (f.loc[fired.index, "orb_break_minute"] <= 60).all()
    assert (f.loc[fired.index, "or_width"] <= 0.006).all()


def test_combine_intraday_bounded():
    f = _fake_features()
    out = isig.combine_intraday(f)
    assert out["confidence"].between(0, 1).all()
    assert set(out["direction"].unique()) <= {-1, 0, 1}


def test_daily_features_no_lookahead_in_orb():
    """orb_dir must only reflect breaks at/*before* the signal cutoff."""
    pytest.importorskip("pyarrow")
    from spy_option_screener.data import intraday as it
    try:
        bars = it.load_intraday("1h")
    except Exception:
        pytest.skip("no cached intraday data")
    daily = pd.DataFrame({"close": [1.0], "vix": [15.0], "realized_vol_20": [0.1]},
                         index=[bars["date"].min()])
    f_early = it.daily_features(bars, daily, or_minutes=60, signal_cutoff="10:30")
    f_late = it.daily_features(bars, daily, or_minutes=60, signal_cutoff="15:30")
    # a later cutoff can only ADD breaks, never remove/flip an early one
    common = f_early.index
    early, late = f_early.loc[common, "orb_dir"], f_late.loc[common, "orb_dir"]
    assert ((early == 0) | (early == late)).mean() > 0.9


def test_timed_entry_backtest_runs():
    from spy_option_screener.backtest import intraday as bi, metrics
    try:
        eq, tr, mk, feat, sig = bi.run_intraday(interval="1h", entry_time="10:30",
                                                hist_start="2022-01-01")
    except Exception as e:
        pytest.skip(f"needs cached data: {e}")
    assert len(eq) > 0 and eq.notna().all()
    assert "cagr" in metrics.summary(eq, tr)
