"""Fast sanity checks: pricing identities, no-look-ahead, backtest runs."""
import numpy as np
import pandas as pd
import pytest

from spy_option_screener.pricing import black_scholes as bs
from spy_option_screener.pricing import vol_model as vm


def test_put_call_parity():
    S, K, t, sig, r, q = 500.0, 500.0, 7 / 365, 0.15, 0.04, 0.013
    c = bs.price(S, K, t, sig, r, q, "call")
    p = bs.price(S, K, t, sig, r, q, "put")
    lhs = c - p
    rhs = S * np.exp(-q * t) - K * np.exp(-r * t)
    assert abs(lhs - rhs) < 1e-6


def test_implied_vol_roundtrip():
    S, K, t = 500.0, 495.0, 5 / 365
    for true_iv in (0.08, 0.15, 0.35):
        px = float(bs.price(S, K, t, true_iv, kind="call"))
        got = bs.implied_vol(px, S, K, t, kind="call")
        assert abs(got - true_iv) < 1e-3


def test_greeks_signs():
    g = bs.greeks(500, 500, 5 / 365, 0.15, kind="call")
    assert 0 < g["delta"] < 1
    assert g["gamma"] > 0
    assert g["theta"] < 0          # long option decays
    assert g["vega"] > 0
    gp = bs.greeks(500, 500, 5 / 365, 0.15, kind="put")
    assert -1 < gp["delta"] < 0


def test_skew_puts_richer_than_calls():
    atm = vm.atm_vol_from_vix(18.0)
    iv_otm_put = vm.skew_adjust(atm, 470, 500, 5)
    iv_otm_call = vm.skew_adjust(atm, 530, 500, 5)
    assert iv_otm_put > iv_otm_call        # equity skew


def test_expected_move_scales_with_sqrt_time():
    m1 = vm.expected_move(500, 0.16, 1)
    m4 = vm.expected_move(500, 0.16, 4)
    assert abs(m4 / m1 - 2.0) < 1e-6


def _fake_history(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    ret = rng.normal(0.0004, 0.011, n)
    close = 400 * np.cumprod(1 + ret)
    df = pd.DataFrame({
        "open": close * (1 - rng.normal(0, 0.002, n)),
        "high": close * (1 + abs(rng.normal(0, 0.004, n))),
        "low": close * (1 - abs(rng.normal(0, 0.004, n))),
        "close": close,
        "volume": rng.integers(5e7, 1e8, n),
        "vix": np.clip(16 + rng.normal(0, 3, n).cumsum() * 0.1, 10, 45),
    }, index=idx)
    df["ret"] = df["close"].pct_change()
    df["realized_vol_20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    return df.dropna()


def test_signals_no_lookahead():
    from spy_option_screener.signals import strategies as st
    df = _fake_history()
    sig = st.combine(df)
    # Truncating the input must not change earlier signal values.
    sig_short = st.combine(df.iloc[:-20])
    common = sig_short.index
    pd.testing.assert_series_equal(
        sig.loc[common, "direction"], sig_short["direction"], check_names=False)


@pytest.mark.parametrize("name", ["pullback", "trend_continuation"])
def test_new_signals_shape_and_no_lookahead(name):
    from spy_option_screener.signals import strategies as st
    df = _fake_history(n=500, seed=3)
    full = st.STRATEGIES[name](df)
    assert set(full["direction"].unique()) <= {-1, 0, 1}
    assert full["confidence"].between(0, 1).all()
    trimmed = st.STRATEGIES[name](df.iloc[:-15])
    pd.testing.assert_series_equal(full.loc[trimmed.index, "direction"],
                                   trimmed["direction"], check_names=False)


def test_pullback_only_longs_dips_in_uptrend():
    """A long-call pullback signal must occur with price above its 50d MA."""
    from spy_option_screener.signals import strategies as st
    from spy_option_screener.signals import indicators as ind
    df = _fake_history(n=600, seed=7)
    sig = st.pullback(df)
    ma50 = ind.sma(df["close"], 50)
    calls = sig.index[sig["direction"] == 1]
    if len(calls):
        assert (df.loc[calls, "close"] > ma50.loc[calls]).mean() > 0.95


def test_backtest_runs():
    from spy_option_screener.signals import strategies as st
    from spy_option_screener.backtest import engine, metrics
    df = _fake_history(n=500)
    sig = st.combine(df)
    equity, trades, marks = engine.run(df, sig, starting_capital=25_000, warmup=210)
    assert len(equity) > 0
    assert equity.notna().all()
    summ = metrics.summary(equity, trades)
    assert "cagr" in summ
