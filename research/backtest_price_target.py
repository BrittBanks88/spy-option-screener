"""Honest hit-rate test for the price-target view's actual numbers.

screener/price_target.py's direction call reuses the two backtested trade-plan
signals, but the near/far targets and watch (cut) level are ATR-scaled
heuristics that had never been checked against what SPY actually did
afterward. This walks every qualifying session forward on daily bars and
asks, plainly:

    - did price touch the near target before the watch level, and in how
      many sessions?
    - did price touch the far target before the watch level?
    - how often did the watch (cut) level get hit first -- i.e. the read
      would have told you to cut?

    python research/backtest_price_target.py

Daily-bar path only (no intraday), reconstructed levels. Not investment advice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spy_option_screener.data import loader, market_calendar as mcal    # noqa: E402
from spy_option_screener.signals import indicators as ind               # noqa: E402
from spy_option_screener.screener import price_target as pt             # noqa: E402

MAX_SESSIONS_AHEAD = 10   # ~2 trading weeks -- generous vs. the "within the
                          # week" far-target label, so we're not the ones
                          # cutting the window short in our own favor


def walk_forward(daily, session_date, direction, near, far, watch):
    entry_date = mcal.next_trading_day(session_date)
    entry_ts = pd.Timestamp(entry_date)
    if entry_ts not in daily.index:
        return None
    path = daily.loc[entry_ts:].iloc[:MAX_SESSIONS_AHEAD]
    if path.empty:
        return None

    near_hit = far_hit = watch_hit = None
    for i, (_, bar) in enumerate(path.iterrows(), start=1):
        lo, hi = bar["low"], bar["high"]
        if watch_hit is None and ((lo <= watch) if direction > 0 else (hi >= watch)):
            watch_hit = i
        if near_hit is None and ((hi >= near) if direction > 0 else (lo <= near)):
            near_hit = i
        if far_hit is None and ((hi >= far) if direction > 0 else (lo <= far)):
            far_hit = i
        if watch_hit is not None and near_hit is not None and far_hit is not None:
            break

    return {
        "date": session_date, "direction": direction,
        "near_first": (near_hit is not None and (watch_hit is None or near_hit <= watch_hit)),
        "near_sessions": near_hit,
        "far_first": (far_hit is not None and (watch_hit is None or far_hit <= watch_hit)),
        "far_sessions": far_hit,
        "watch_first": (watch_hit is not None and (near_hit is None or watch_hit < near_hit)),
    }


def main():
    print("Refreshing SPY/VIX history...")
    daily = loader.load_history(start="2018-01-01", refresh=True)
    daily.index = pd.DatetimeIndex(daily.index)
    atr14 = ind.atr(daily["high"], daily["low"], daily["close"], 14)

    rows = []
    for i in range(210, len(daily) - 1):
        session_date = daily.index[i].date()
        dd = daily.iloc[:i + 1]
        direction, confidence, reasons, warning = pt._blend(dd)
        if direction == 0 or confidence < pt.MIN_CONFIDENCE:
            continue
        spot = float(dd["close"].iloc[-1])
        atr = float(atr14.iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
        near = spot + direction * 1.0 * atr
        if direction > 0:
            swing = float(dd["high"].tail(20).max())
            far = min(max(swing, spot + 1.8 * atr), spot + 2.8 * atr)
        else:
            swing = float(dd["low"].tail(20).min())
            far = max(min(swing, spot - 1.8 * atr), spot - 2.8 * atr)
        watch = spot - direction * 1.0 * atr

        r = walk_forward(daily, session_date, direction, near, far, watch)
        if r is None:
            continue
        r["regime_flagged"] = bool(warning)
        rows.append(r)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    if df.empty:
        print("no qualifying sessions")
        return

    def report(label, g):
        if g.empty:
            print(f"{label}: n=0")
            return
        near_sess = g.loc[g["near_first"], "near_sessions"]
        far_sess = g.loc[g["far_first"], "far_sessions"]
        print(f"{label}: n={len(g)}")
        print(f"  near target hit first: {g['near_first'].mean():.0%}"
              + (f"  (median {near_sess.median():.0f} sessions)" if len(near_sess) else ""))
        print(f"  far target hit first:  {g['far_first'].mean():.0%}"
              + (f"  (median {far_sess.median():.0f} sessions)" if len(far_sess) else ""))
        print(f"  watch/cut level hit first (would've said cut): {g['watch_first'].mean():.0%}")

    print(f"\n{'='*90}\nFULL HISTORY (2018-2026)\n{'='*90}")
    report("All qualifying reads", df)
    print()
    report("  bullish (calls lean)", df[df.direction > 0])
    report("  bearish (puts lean)", df[df.direction < 0])
    print()
    report("  bearish, regime-flagged (uptrend not yet confirmed bearish)",
           df[(df.direction < 0) & df.regime_flagged])
    report("  bearish, NOT flagged (confirmed downtrend)",
           df[(df.direction < 0) & ~df.regime_flagged])

    print(f"\n{'='*90}\n2024-2026 ONLY\n{'='*90}")
    recent = df[df["date"] >= "2024-01-01"]
    report("All qualifying reads", recent)
    report("  bullish (calls lean)", recent[recent.direction > 0])
    report("  bearish (puts lean)", recent[recent.direction < 0])


if __name__ == "__main__":
    main()
