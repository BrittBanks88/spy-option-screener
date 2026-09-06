"""Vectorized technical indicators. All take a price/close Series, return a Series
aligned to the same index. No look-ahead: every value uses only data up to and
including that bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI. n=2 is the classic short-term mean-reversion setting."""
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def macd(s: pd.Series, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def roc(s: pd.Series, n: int = 5) -> pd.Series:
    return s.pct_change(n)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(s, n)
    sd = s.rolling(n).std()
    return mid - k * sd, mid, mid + k * sd


def zscore(s: pd.Series, n: int = 60) -> pd.Series:
    mean = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return (s - mean) / sd.replace(0.0, np.nan)


def donchian(high: pd.Series, low: pd.Series, n: int = 20):
    return low.rolling(n).min(), high.rolling(n).max()


def pct_rank(s: pd.Series, n: int = 252) -> pd.Series:
    """Rolling percentile rank of the latest value within the trailing window
    (0..1). Used for IV Rank / realized-vol rank style features.
    """
    return s.rolling(n).apply(
        lambda w: (w <= w[-1]).mean(), raw=True
    )
