"""Glue: run the option backtest on same-day intraday signals with a timed entry.

Ties together data.intraday (features), signals.intraday_signals (gap /
opening-range-breakout) and backtest.engine (the option simulator), so a caller
just picks an interval, an entry time and a signal set.
"""
from __future__ import annotations

import pandas as pd

from ..data import intraday as it
from ..data import loader
from ..signals import intraday_signals as isig
from ..signals import strategies as dsig
from .engine import run as engine_run
from .rules import TradeRules


def _entry_price_series(feat: pd.DataFrame, entry_time: str) -> pd.Series:
    col = f"px_{entry_time.replace(':', '')}"
    if col not in feat.columns:
        raise ValueError(f"no precomputed price column {col!r}; add {entry_time} "
                         "to daily_features(entry_times=...)")
    return feat[col].dropna()


def run_intraday(interval="1h", entry_time="10:30", or_minutes=None,
                 signal_names=("gap_continuation", "opening_range_breakout"),
                 use_daily_bias=True, rules: TradeRules | None = None,
                 starting_capital=25_000.0, hist_start="2018-01-01",
                 refresh=False):
    """Return (equity, trades, marks, feat, signal)."""
    or_minutes = or_minutes or (60 if interval == "1h" else 30)
    entry_times = tuple(sorted({entry_time, "10:00", "10:30", "11:00"}))

    daily = loader.load_history(start=hist_start, refresh=refresh)
    bars = it.load_intraday(interval, refresh=refresh)
    feat = it.daily_features(bars, daily, or_minutes=or_minutes,
                             signal_cutoff=entry_time, entry_times=entry_times)

    daily_bias = dsig.combine(daily) if use_daily_bias else None
    if daily_bias is not None:
        daily_bias.index = pd.DatetimeIndex(daily_bias.index)

    signal = isig.combine_intraday(feat, names=list(signal_names),
                                   daily_bias=daily_bias)

    # align everything to the intraday feature dates
    dd = daily.copy()
    dd.index = pd.DatetimeIndex(dd.index)
    dd = dd.reindex(feat.index).dropna(subset=["close", "vix"])
    sig = signal.reindex(dd.index)
    entry_px = _entry_price_series(feat, entry_time).reindex(dd.index)

    rules = rules or TradeRules(entry_time=entry_time)
    warmup = min(60, max(0, len(dd) // 4))
    equity, trades, marks = engine_run(
        dd, sig, rules, starting_capital=starting_capital, warmup=warmup,
        entry_prices=entry_px, same_day_signal=True)
    return equity, trades, marks, feat, sig
