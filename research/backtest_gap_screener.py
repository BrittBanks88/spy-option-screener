"""Full simulation of the exact gap-screener rule: for every qualifying SPY
gap since 2000, walk forward day by day and record whichever of
TARGET / STOP / TIME-OUT happens first -- using the precise levels
screener/gap_screener.py computes (stop = 2.0x ATR beyond the gap open,
target = the pre-gap level, time-stop = 14 trading days).

    python research/backtest_gap_screener.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spy_option_screener.data import loader                              # noqa: E402
from spy_option_screener.signals import indicators as ind                 # noqa: E402
from spy_option_screener.screener.gap_screener import (                   # noqa: E402
    MIN_GAP_PCT, STOP_ATR_MULT, TIME_STOP_DAYS,
)


def main():
    daily = loader.load_history(start="2000-01-01")
    daily.index = pd.DatetimeIndex(daily.index)
    atr14 = ind.atr(daily["high"], daily["low"], daily["close"], 14)
    prior_close = daily["close"].shift(1)

    rows = []
    for i in range(60, len(daily)):
        gap_pct = (daily["open"].iloc[i] - prior_close.iloc[i]) / prior_close.iloc[i]
        if not np.isfinite(gap_pct) or abs(gap_pct) < MIN_GAP_PCT:
            continue
        direction = 1 if gap_pct > 0 else -1
        target = float(prior_close.iloc[i])
        gap_open = float(daily["open"].iloc[i])
        atr = float(atr14.iloc[i - 1])
        stop = gap_open + direction * STOP_ATR_MULT * atr

        outcome, day_hit = "timeout", TIME_STOP_DAYS
        end = min(i + TIME_STOP_DAYS + 1, len(daily))
        for j in range(i, end):
            hi, lo = daily["high"].iloc[j], daily["low"].iloc[j]
            hit_target = (lo <= target) if direction > 0 else (hi >= target)
            hit_stop = (hi >= stop) if direction > 0 else (lo <= stop)
            if hit_target and hit_stop:
                # same-day ambiguity -- resolve conservatively: assume the
                # WORSE outcome (stop) unless the open itself already cleared
                # the target (gap-through), matching how you'd actually manage it
                o = daily["open"].iloc[j]
                cleared = (o <= target) if direction > 0 else (o >= target)
                outcome, day_hit = ("target", j - i) if cleared else ("stop", j - i)
                break
            if hit_target:
                outcome, day_hit = "target", j - i
                break
            if hit_stop:
                outcome, day_hit = "stop", j - i
                break
        rows.append({"date": daily.index[i], "direction": direction,
                     "gap_pct": gap_pct, "outcome": outcome, "day_hit": day_hit})

    df = pd.DataFrame(rows)
    print(f"n={len(df)} qualifying gaps, {df.date.min().date()} -> {df.date.max().date()}\n")
    counts = df["outcome"].value_counts()
    for o in ["target", "stop", "timeout"]:
        n = counts.get(o, 0)
        med = df.loc[df.outcome == o, "day_hit"].median()
        print(f"  {o:<8} {n:>5d}  {n/len(df):>5.0%}   median day {med:.0f}")

    print(f"\nBy direction:")
    for d, name in [(1, "Gap UP"), (-1, "Gap DOWN")]:
        sub = df[df.direction == d]
        c = sub["outcome"].value_counts()
        print(f"  {name}: n={len(sub)}  target {c.get('target',0)/len(sub):.0%}  "
              f"stop {c.get('stop',0)/len(sub):.0%}  "
              f"timeout {c.get('timeout',0)/len(sub):.0%}")

    print(f"\nRecent regime (2018-2026):")
    recent = df[df.date >= "2018-01-01"]
    c = recent["outcome"].value_counts()
    print(f"  n={len(recent)}  target {c.get('target',0)/len(recent):.0%}  "
          f"stop {c.get('stop',0)/len(recent):.0%}  "
          f"timeout {c.get('timeout',0)/len(recent):.0%}")


if __name__ == "__main__":
    main()
