"""Directional signal generators for long SPY option trades.

Each strategy consumes the daily history DataFrame from data.loader and returns
a DataFrame with, per date:

    direction   : +1 (buy calls), -1 (buy puts), 0 (stand aside)
    confidence  : 0..1, how strong the setup is (used for position sizing / ranking)
    edge_note   : short human-readable reason

Signals are computed on the close and are actionable at the NEXT bar's open,
so the backtester must shift entries by one day. Nothing here peeks ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def _frame(index) -> pd.DataFrame:
    return pd.DataFrame({
        "direction": pd.Series(0, index=index, dtype=int),
        "confidence": pd.Series(0.0, index=index, dtype=float),
        "edge_note": pd.Series("", index=index, dtype=object),
    })


def momentum_breakout(df: pd.DataFrame, lookback=20, trend=100, roc_n=5) -> pd.DataFrame:
    """Trend-following: trade in the direction of a fresh N-day range break that
    agrees with the long-term trend. Historically the more reliable long-option
    setup on SPY because breakouts often precede vol expansion.
    """
    out = _frame(df.index)
    close = df["close"]
    lo, hi = ind.donchian(df["high"], df["low"], lookback)
    trend_ma = ind.sma(close, trend)
    momentum = ind.roc(close, roc_n)

    up = (close > hi.shift(1)) & (close > trend_ma)
    dn = (close < lo.shift(1)) & (close < trend_ma)

    # Confidence: how far the ROC is stretched, capped.
    conf = (momentum.abs() / 0.03).clip(0.0, 1.0)

    out.loc[up, "direction"] = 1
    out.loc[dn, "direction"] = -1
    out.loc[up | dn, "confidence"] = conf[up | dn]
    out.loc[up, "edge_note"] = "20d breakout up, above 100d MA"
    out.loc[dn, "edge_note"] = "20d breakdown, below 100d MA"
    return out


def mean_reversion(df: pd.DataFrame, rsi_n=2, lo_th=5.0, hi_th=95.0,
                   trend=200) -> pd.DataFrame:
    """RSI(2) snapback. Buy calls when very oversold in an uptrend; buy puts
    when very overbought in a downtrend. Caveat baked into confidence: these
    setups usually come with elevated IV, so the screener will still penalize
    expensive contracts downstream.
    """
    out = _frame(df.index)
    close = df["close"]
    r = ind.rsi(close, rsi_n)
    trend_ma = ind.sma(close, trend)

    long_c = (r < lo_th) & (close > trend_ma)
    long_p = (r > hi_th) & (close < trend_ma)

    conf_c = ((lo_th - r) / lo_th).clip(0.0, 1.0)
    conf_p = ((r - hi_th) / (100.0 - hi_th)).clip(0.0, 1.0)

    out.loc[long_c, "direction"] = 1
    out.loc[long_c, "confidence"] = conf_c[long_c]
    out.loc[long_c, "edge_note"] = "RSI(2) oversold in uptrend"
    out.loc[long_p, "direction"] = -1
    out.loc[long_p, "confidence"] = conf_p[long_p]
    out.loc[long_p, "edge_note"] = "RSI(2) overbought in downtrend"
    return out


def vol_regime(df: pd.DataFrame, z_n=60, cheap_z=-1.0) -> pd.DataFrame:
    """Convexity play: when implied vol (VIX) is unusually low vs its own recent
    range AND realized vol is ticking up, long premium is cheap relative to the
    move that may be coming. Direction follows the 20d trend so it's not a
    naked straddle.
    """
    out = _frame(df.index)
    vix_z = ind.zscore(df["vix"], z_n)
    rv = df["realized_vol_20"]
    rv_rising = rv > rv.shift(5)
    trend_ma = ind.sma(df["close"], 20)

    cheap = (vix_z < cheap_z) & rv_rising
    dir_ = np.where(df["close"] > trend_ma, 1, -1)

    out.loc[cheap, "direction"] = dir_[cheap.values]
    out.loc[cheap, "confidence"] = (cheap_z - vix_z).clip(0.0, 2.0)[cheap] / 2.0
    out.loc[cheap, "edge_note"] = "IV cheap vs range, realized vol rising"
    return out


def pullback(df: pd.DataFrame, fast=10, trend=50, deep=200, rsi_n=4,
             rsi_th=35.0, max_dip=0.05) -> pd.DataFrame:
    """Buy the dip inside an established trend.

    Long calls when: price is in an uptrend (rising 50d MA, 50d above 200d),
    has pulled back (short RSI oversold or a tag of the fast MA) but NOT broken
    the trend (still above the 50d, dip shallower than ``max_dip``), and has
    just shown a stabilising up-day. Mirror image for downtrends -> puts.

    This differs from ``mean_reversion`` (pure RSI-2 extreme, any context): here
    the pullback must be a shallow, trend-preserving one.
    """
    out = _frame(df.index)
    close = df["close"]
    ma_f, ma_t, ma_d = ind.sma(close, fast), ind.sma(close, trend), ind.sma(close, deep)
    r = ind.rsi(close, rsi_n)
    up_day = close > close.shift(1)
    dn_day = close < close.shift(1)

    swing_hi = close.rolling(trend).max()
    swing_lo = close.rolling(trend).min()
    dip = (swing_hi - close) / swing_hi          # how far below the recent high
    pop = (close - swing_lo) / swing_lo

    up_trend = (ma_t > ma_t.shift(fast)) & (ma_t > ma_d) & (close > ma_t)
    down_trend = (ma_t < ma_t.shift(fast)) & (ma_t < ma_d) & (close < ma_t)

    # a genuine pullback: short-RSI soft AND price stretched below the fast MA
    pulled_c = (r < rsi_th) | ((close < ma_f) & (r < 45))
    pulled_p = (r > 100 - rsi_th) | ((close > ma_f) & (r > 55))
    bounced_up = (r > r.shift(1)) | up_day
    faded_down = (r < r.shift(1)) | dn_day

    long_c = up_trend & pulled_c.shift(1).fillna(False) & bounced_up & (dip <= max_dip)
    long_p = down_trend & pulled_p.shift(1).fillna(False) & faded_down & (pop <= max_dip)

    conf_c = (0.35 + 0.65 * ((rsi_th - r.shift(1)) / rsi_th)).clip(0.0, 1.0)
    conf_p = (0.35 + 0.65 * ((r.shift(1) - (100 - rsi_th)) / rsi_th)).clip(0.0, 1.0)

    out.loc[long_c, "direction"] = 1
    out.loc[long_c, "confidence"] = conf_c[long_c].fillna(0.4)
    out.loc[long_c, "edge_note"] = "pullback in an uptrend, turning up"
    out.loc[long_p, "direction"] = -1
    out.loc[long_p, "confidence"] = conf_p[long_p].fillna(0.4)
    out.loc[long_p, "edge_note"] = "bounce in a downtrend, rolling over"
    return out


def trend_continuation(df: pd.DataFrame, s1=20, s2=50, s3=100, breakout=10,
                       max_ext_atr=2.5) -> pd.DataFrame:
    """Ride an established trend as it makes a fresh leg.

    Long calls when the moving-average stack is bullish and rising
    (20d > 50d > 100d, all sloping up), price is not blown off (within
    ``max_ext_atr`` ATRs of the 20d MA), and today prints a new ``breakout``-day
    high after a short pause. Mirror image for downtrends -> puts.
    """
    out = _frame(df.index)
    close = df["close"]
    m1, m2, m3 = ind.sma(close, s1), ind.sma(close, s2), ind.sma(close, s3)
    atr = ind.atr(df["high"], df["low"], close, 14)

    stack_up = (m1 > m2) & (m2 > m3) & (m1 > m1.shift(5)) & (m2 > m2.shift(10))
    stack_dn = (m1 < m2) & (m2 < m3) & (m1 < m1.shift(5)) & (m2 < m2.shift(10))
    ext = (close - m1).abs() / atr.replace(0, np.nan)

    hi_n = close.rolling(breakout).max()
    lo_n = close.rolling(breakout).min()
    paused_up = (close.shift(1) < hi_n.shift(2))     # wasn't already at highs yesterday
    paused_dn = (close.shift(1) > lo_n.shift(2))

    long_c = stack_up & (close >= hi_n.shift(1)) & paused_up & (ext <= max_ext_atr)
    long_p = stack_dn & (close <= lo_n.shift(1)) & paused_dn & (ext <= max_ext_atr)

    spread = ((m1 - m3).abs() / close)
    conf = (0.4 + spread / 0.04).clip(0.0, 1.0)

    out.loc[long_c, "direction"] = 1
    out.loc[long_c, "confidence"] = conf[long_c]
    out.loc[long_c, "edge_note"] = "uptrend intact, new leg up"
    out.loc[long_p, "direction"] = -1
    out.loc[long_p, "confidence"] = conf[long_p]
    out.loc[long_p, "edge_note"] = "downtrend intact, new leg down"
    return out


STRATEGIES = {
    "momentum_breakout": momentum_breakout,
    "mean_reversion": mean_reversion,
    "pullback": pullback,
    "trend_continuation": trend_continuation,
    "vol_regime": vol_regime,
}


def combine(df: pd.DataFrame, names=None, weights=None) -> pd.DataFrame:
    """Blend several strategies. Directions are summed with weights; the net
    sign is the trade direction and |net| (normalized) feeds confidence.
    """
    names = names or list(STRATEGIES)
    weights = weights or {n: 1.0 for n in names}
    sigs = {n: STRATEGIES[n](df) for n in names}

    score = sum(weights[n] * sigs[n]["direction"] * sigs[n]["confidence"]
                for n in names)
    total_w = sum(abs(weights[n]) for n in names)

    out = _frame(df.index)
    out["direction"] = np.sign(score).astype(int)
    out["confidence"] = (score.abs() / max(total_w, 1e-9)).clip(0.0, 1.0)
    # Attach the note from the highest-weighted contributing strategy.
    for n in names:
        active = (sigs[n]["direction"] != 0) & (out["direction"] == np.sign(score).astype(int))
        out.loc[active & (out["edge_note"] == ""), "edge_note"] = sigs[n].loc[active, "edge_note"]
    return out
