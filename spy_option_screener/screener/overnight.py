"""Overnight-hold assessment for long SPY calls/puts.

The honest finding first (see research/backtest_overnight.py): buying an option
purely to hold ONE night loses money -- the bid/ask + a night of theta dwarf
the ~3-5 bps of overnight index drift. So this module does NOT emit a
stand-alone "buy overnight" signal.

What it DOES: rate how friendly tonight is for *carrying a position you already
hold* (e.g. the pullback->continuation trade) through the close. Drift is
slightly with you when you're long calls in an uptrend on a Mon/Tue night with
a calm VIX; it's against you into a Friday, a holiday, or a VIX spike.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import market_calendar as mcal
from ..signals import indicators as ind


@dataclass
class OvernightView:
    rating: str            # "favourable" | "neutral" | "avoid"
    score: float           # -1 .. +1 (sign = with/against a long call)
    exp_move_pct: float    # 1-sigma close->open move, recent
    drift_bps: float       # historical mean overnight drift in this regime
    calendar_nights: int   # calendar days of decay you carry (Fri = 3)
    note: str

    def line(self, direction: int = 1) -> str:
        side = "calls" if direction > 0 else "puts"
        return (f"Overnight ({self.calendar_nights}-night carry): "
                f"*{self.rating}* for {side} — {self.note} "
                f"(±{self.exp_move_pct:.2%} typical gap, drift {self.drift_bps:+.1f} bps).")


def assess(daily: pd.DataFrame, as_of: dt.date | pd.Timestamp | None = None,
           direction: int = 1) -> OvernightView:
    d = daily.copy()
    d.index = pd.DatetimeIndex(d.index)
    if as_of is not None:
        d = d.loc[:pd.Timestamp(as_of)]
    row = d.iloc[-1]
    close_date = d.index[-1].date()
    next_sess = mcal.next_trading_day(close_date)
    nights = (next_sess - close_date).days

    on = (d["open"].shift(-1) / d["close"] - 1.0).dropna()
    one_night = float(on.tail(40).std()) if len(on) > 40 else 0.004
    exp_move = one_night * np.sqrt(max(nights, 1))
    ma50 = float(ind.sma(d["close"], 50).iloc[-1])
    day_ret = float(row["close"] / row["open"] - 1.0)
    vix = float(row["vix"])
    vix_chg = float(vix - d["vix"].iloc[-2]) if len(d) > 1 else 0.0
    dow = next_sess.weekday()      # 0 Mon .. 4 Fri  (session the carry lands in)

    # regime drift: overnight mean when long-trend + calm, vs stressed
    trend_up = row["close"] > ma50
    calm = vix < 20 and vix_chg < 1.0
    hist = on.tail(750)
    drift = float(hist.mean() * 1e4) if len(hist) else 3.0

    score = 0.0
    bits = []
    if trend_up and direction > 0:
        score += 0.35; bits.append("uptrend")
    elif (not trend_up) and direction < 0:
        score += 0.35; bits.append("downtrend")
    else:
        score -= 0.25; bits.append("trend not with you")
    if calm:
        score += 0.25; bits.append("VIX calm")
    if vix > 22 or vix_chg > 1.5:
        score -= 0.35; bits.append("VIX elevated/rising")
    if nights == 1 and dow in (1, 2):
        score += 0.15; bits.append("into Tue/Wed")
    if nights >= 2:
        score -= 0.15 * (nights - 1)
        bits.append(f"{nights} nights of decay")
    if abs(day_ret) > 0.012:
        score -= 0.20; bits.append("big range day tends to revert")

    score = float(np.clip(score, -1, 1))
    rating = ("favourable" if score >= 0.35 else
              "avoid" if score <= -0.2 else "neutral")
    return OvernightView(
        rating=rating, score=score, exp_move_pct=exp_move, drift_bps=drift,
        calendar_nights=nights, note=", ".join(bits) or "mixed",
    )
