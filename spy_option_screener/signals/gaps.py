"""Shared resolution rule for a gap-fill trade: which of TARGET or STOP is
touched first.

Used by both the live screener and the tracker so they can never disagree.
Same-bar ambiguity (both levels touched inside one daily bar) resolves
conservatively -- the STOP wins, unless the bar OPENED already through the
target (a gap straight over it). That is the exact rule the 85/14/1 backtest
(research/backtest_gap_screener.py) was scored with, so live accuracy is
comparable to the published numbers instead of quietly optimistic.
"""
from __future__ import annotations

import pandas as pd


def first_resolution(path: pd.DataFrame, direction: int, target: float,
                     stop: float) -> tuple[str | None, pd.Timestamp | None]:
    """Walk ``path`` (daily bars, oldest first). direction: +1 = gap UP
    (target below, stop above), -1 = gap DOWN (target above, stop below).
    Returns ("target" | "stop", bar timestamp) for the first level touched,
    or (None, None) if neither has been."""
    for ts, bar in path.iterrows():
        if direction > 0:
            hit_target, hit_stop = bar["low"] <= target, bar["high"] >= stop
            opened_through = bar["open"] <= target
        else:
            hit_target, hit_stop = bar["high"] >= target, bar["low"] <= stop
            opened_through = bar["open"] >= target
        if hit_target and hit_stop:
            return ("target" if opened_through else "stop"), ts
        if hit_target:
            return "target", ts
        if hit_stop:
            return "stop", ts
    return None, None
