"""Do SPY gaps actually fill within 7-14 days? Tested on 26+ years of data.

A "gap" here = today's open vs. yesterday's close (an overnight move with no
trading in between). "Filled" = price has traded back through that prior
close level (checked from the gap day itself onward, so a same-day fill
counts as day 0). Reports the full time-to-fill distribution, bucketed by
gap size and by whether the gap is WITH or AGAINST the prevailing trend,
so we can see honestly whether "fills in 7-14 days" describes most gaps,
a specific subset of them, or isn't really true at all.

    python research/backtest_gaps.py

Not investment advice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spy_option_screener.data import loader                 # noqa: E402
from spy_option_screener.signals import indicators as ind    # noqa: E402

MAX_HORIZON = 60          # trading days to search for a fill before giving up
MIN_GAP_PCT = 0.0015      # ~0.15% -- filter out pure noise, keep everything else


def find_gaps(daily: pd.DataFrame) -> pd.DataFrame:
    prior_close = daily["close"].shift(1)
    gap_pct = (daily["open"] - prior_close) / prior_close
    atr = ind.atr(daily["high"], daily["low"], daily["close"], 14)
    sma50 = ind.sma(daily["close"], 50)

    rows = []
    for i in range(60, len(daily)):
        g = gap_pct.iloc[i]
        if not np.isfinite(g) or abs(g) < MIN_GAP_PCT:
            continue
        direction = 1 if g > 0 else -1          # +1 = gap up, -1 = gap down
        level = float(prior_close.iloc[i])       # the level that needs to be re-touched
        trend_up = daily["close"].iloc[i - 1] > sma50.iloc[i - 1]
        with_trend = (direction > 0 and trend_up) or (direction < 0 and not trend_up)

        fill_day = None
        for j in range(i, min(i + MAX_HORIZON + 1, len(daily))):
            lo, hi = daily["high"].iloc[j], daily["low"].iloc[j]
            hit = (daily["low"].iloc[j] <= level) if direction > 0 else (daily["high"].iloc[j] >= level)
            if hit:
                fill_day = j - i
                break
        rows.append({
            "date": daily.index[i], "direction": direction, "gap_pct": g,
            "atr_pct": float(atr.iloc[i - 1]) / float(daily["close"].iloc[i - 1]),
            "with_trend": with_trend, "fill_day": fill_day,
        })
    return pd.DataFrame(rows)


def bucket_fill_day(d):
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "never (60d+)"
    d = int(d)
    if d == 0:
        return "same day"
    if d <= 2:
        return "1-2 days"
    if d <= 6:
        return "3-6 days"
    if d <= 14:
        return "7-14 days"
    if d <= 30:
        return "15-30 days"
    return "31-60 days"


BUCKET_ORDER = ["same day", "1-2 days", "3-6 days", "7-14 days",
                "15-30 days", "31-60 days", "never (60d+)"]


def report(label, g: pd.DataFrame):
    n = len(g)
    if n == 0:
        print(f"{label}: n=0")
        return
    counts = g["fill_day"].map(bucket_fill_day).value_counts()
    counts = counts.reindex(BUCKET_ORDER, fill_value=0)
    filled = g["fill_day"].notna()
    print(f"{label}: n={n}  ·  ever filled within 60d: {filled.mean():.0%}  ·  "
          f"median fill (when it fills): {g.loc[filled,'fill_day'].median():.0f}d")
    for b in BUCKET_ORDER:
        pct = counts[b] / n
        bar = "#" * int(pct * 40)
        print(f"    {b:<14} {counts[b]:>4d}  {pct:>5.0%}  {bar}")


def main():
    print("Loading full SPY history...")
    daily = loader.load_history(start="2000-01-01", refresh=True)
    daily.index = pd.DatetimeIndex(daily.index)
    gaps = find_gaps(daily)
    print(f"{len(gaps)} qualifying gaps (|gap| >= {MIN_GAP_PCT:.2%}) out of "
          f"{len(daily)} sessions, {daily.index[0].date()} -> {daily.index[-1].date()}\n")

    print("=" * 78 + "\nALL GAPS\n" + "=" * 78)
    report("All gaps", gaps)
    print()
    report("  Gap UP", gaps[gaps.direction > 0])
    report("  Gap DOWN", gaps[gaps.direction < 0])
    print()
    report("  WITH the trend (continuation)", gaps[gaps.with_trend])
    report("  AGAINST the trend (reversal-ish)", gaps[~gaps.with_trend])

    print("\n" + "=" * 78 + "\nBY GAP SIZE (percentile terciles of |gap_pct|)\n" + "=" * 78)
    abs_gap = gaps["gap_pct"].abs()
    q1, q2 = abs_gap.quantile([1/3, 2/3])
    small = gaps[abs_gap <= q1]
    medium = gaps[(abs_gap > q1) & (abs_gap <= q2)]
    large = gaps[abs_gap > q2]
    print(f"tercile cutoffs: small <= {q1:.2%}  ·  medium <= {q2:.2%}  ·  large > {q2:.2%}\n")
    report(f"  Small gaps", small)
    report(f"  Medium gaps", medium)
    report(f"  Large gaps", large)

    print("\n" + "=" * 78 + "\nMEDIUM gaps split by trend (the most likely candidate for a clean\n"
          "7-14 day mean-reversion signal, if one exists)\n" + "=" * 78)
    report("  Medium, WITH trend", medium[medium.with_trend])
    report("  Medium, AGAINST trend", medium[~medium.with_trend])

    print("\n" + "=" * 78 + "\nLARGE gaps split by trend\n" + "=" * 78)
    report("  Large, WITH trend", large[large.with_trend])
    report("  Large, AGAINST trend", large[~large.with_trend])

    out = Path(__file__).resolve().parents[1] / "outputs" / "gap_backtest.csv"
    out.parent.mkdir(exist_ok=True)
    gaps.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
