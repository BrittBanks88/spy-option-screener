"""SPY gap-fill screener -- price target, timing, and stop, no contract mechanics.

A "gap" = today's open vs. yesterday's close (an overnight move with no
trading in between). Backtested on 26+ years of SPY daily data
(research/backtest_gaps.py, n=4,776 qualifying gaps since 2000):

    - 89% of all qualifying gaps fill (price re-touches the pre-gap level)
      within 14 trading days. Even the toughest slice (large gaps) clears 82%.
    - Gaps that DO fill run, on the way there, a median of only 0.36 ATR
      beyond the gap open (95th pct: 1.9 ATR) before turning back.
    - Gaps that DON'T fill within 14 days run a median of 3.4 ATR beyond the
      gap open over that window -- a starkly different signature.
    - A stop at 2.0x ATR beyond the gap open cuts only ~4% of eventual
      winners early, while catching ~91% of the ones that were never going
      to fill in time -- that's what STOP is calibrated to below.

TARGET = the pre-gap price level. STOP = 2.0x ATR beyond the gap open, away
from target. TIME-STOP = 14 trading days from the gap. Informational only,
not a trade recommendation.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..data import loader
from ..data import market_calendar as mcal
from ..signals import indicators as ind

ET = ZoneInfo("America/New_York")
MIN_GAP_PCT = 0.0015     # ~0.15% -- filters noise, matches research/backtest_gaps.py
STOP_ATR_MULT = 2.0
TIME_STOP_DAYS = 14


@dataclass
class GapSignal:
    qualified: bool
    asof: str
    day_note: str
    gap_date: str = ""
    gap_type: str = ""             # "up" | "down"
    spot: float = 0.0
    spot_estimated: bool = False
    gap_open: float = 0.0
    target: float = 0.0
    stop: float = 0.0
    days_elapsed: int = 0
    days_left: int = 0
    time_stop_date: str = ""
    historical_fill_rate: float = 0.0
    setup: str = ""
    caveat: str = ("Backtested on 26+ years of SPY daily bars (2000-2026, "
                   "n=4,776 gaps): 89% fill within 14 trading days overall; "
                   "the stop is calibrated to cut ~4% of eventual winners "
                   "early while catching ~91% of the ones that wouldn't have "
                   "filled in time. Reconstructed from daily OHLC, not a "
                   "live quote engine. Informational only.")

    @property
    def target_pct(self) -> float:
        return (self.target / self.spot - 1.0) if self.spot else 0.0

    @property
    def stop_pct(self) -> float:
        return (self.stop / self.spot - 1.0) if self.spot else 0.0

    def slack_text(self) -> str:
        if not self.qualified:
            return (f"📊 *SPY Gap Screener*\n_{self.day_note}_\n\n"
                    f"No open gap signal right now. {self.setup}")
        arrow = "closing down to" if self.gap_type == "up" else "rising up to"
        return "\n".join([
            f"📊 *SPY Gap Screener*  ·  spot ${self.spot:.2f}   _{self.day_note}_",
            "",
            f"*Gap:* {self.gap_type.upper()} on {self.gap_date}  ·  "
            f"day {self.days_elapsed} of {TIME_STOP_DAYS}",
            f"*Target:* ${self.target:.2f}  ({self.target_pct:+.1%})  ·  {arrow} "
            f"the pre-gap level",
            f"*Stop:*  ${self.stop:.2f}  ({self.stop_pct:+.1%})  ·  cuts the "
            f"read if price runs this far the wrong way",
            f"*Time stop:*  {self.time_stop_date}  ({self.days_left} sessions left)",
            f"*Historical fill rate for this type:*  {self.historical_fill_rate:.0%} "
            f"within {TIME_STOP_DAYS} days",
            "",
            f"_{self.caveat}_",
        ])

    def slack_blocks(self) -> list[dict]:
        if not self.qualified:
            return [{"type": "header", "text": {"type": "plain_text",
                     "text": "SPY Gap Screener"}},
                    {"type": "section", "text": {"type": "mrkdwn",
                     "text": f"No open gap signal right now. {self.setup}\n"
                             f"_{self.day_note}_"}}]
        fields = [
            ("Gap", f"{self.gap_type.upper()} · {self.gap_date}\nday "
                    f"{self.days_elapsed}/{TIME_STOP_DAYS}"),
            ("Target", f"${self.target:.2f}\n{self.target_pct:+.1%}"),
            ("Stop", f"${self.stop:.2f}\n{self.stop_pct:+.1%}"),
            ("Time stop", f"{self.time_stop_date}\n{self.days_left} sessions left"),
        ]
        return [
            {"type": "header", "text": {"type": "plain_text",
             "text": f"SPY Gap Screener · {self.gap_type.upper()}"}},
            {"type": "context", "elements": [{"type": "mrkdwn",
             "text": f"spot ${self.spot:.2f} · {self.day_note}"}]},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*{k}*\n{v}"} for k, v in fields]},
            {"type": "section", "text": {"type": "mrkdwn",
             "text": f"Historical fill rate: {self.historical_fill_rate:.0%} "
                     f"within {TIME_STOP_DAYS} days"}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": self.caveat}]},
        ]

    def slack_payload(self) -> dict:
        first = self.slack_text().split("\n\n")[0].replace("*", "")
        return {"text": first, "blocks": self.slack_blocks()}


def _fill_rate_for(gap_pct: float, direction: int) -> float:
    """Coarse lookup matching research/backtest_gaps.py's size/direction
    buckets, so the reported rate matches the specific kind of gap, not
    just the overall 89% blend."""
    if direction > 0:      # gap up
        return 0.86 if abs(gap_pct) > 0.0058 else 0.91
    return 0.92 if abs(gap_pct) > 0.0058 else 0.93


def build_view(hist_start: str = "2000-01-01", refresh: bool = False,
              for_date: str | dt.date | None = None) -> GapSignal:
    daily = loader.load_history(start=hist_start, refresh=refresh)
    daily.index = pd.DatetimeIndex(daily.index)
    atr14 = ind.atr(daily["high"], daily["low"], daily["close"], 14)

    now = dt.datetime.now(ET)
    session_date = (pd.Timestamp(for_date).date() if for_date is not None
                    else daily.index[-1].date())
    asof = now.strftime("%Y-%m-%d %H:%M %Z")
    day_note = (f"As of {session_date:%a %b %d} close" if for_date is not None
                else f"as of {daily.index[-1]:%a %b %d} close")

    idx = daily.index.get_loc(pd.Timestamp(session_date))
    prior_close = daily["close"].shift(1)

    # look back up to TIME_STOP_DAYS for the most recent still-open gap
    for lookback in range(0, TIME_STOP_DAYS + 1):
        i = idx - lookback
        if i < 1:
            break
        gap_pct = (daily["open"].iloc[i] - prior_close.iloc[i]) / prior_close.iloc[i]
        if not np.isfinite(gap_pct) or abs(gap_pct) < MIN_GAP_PCT:
            continue
        direction = 1 if gap_pct > 0 else -1
        target = float(prior_close.iloc[i])
        gap_open = float(daily["open"].iloc[i])
        atr = float(atr14.iloc[i - 1])
        stop = gap_open + direction * STOP_ATR_MULT * atr

        # has it already filled, or hit its stop, between the gap day and today?
        path = daily.iloc[i:idx + 1]
        filled = ((path["low"] <= target).any() if direction > 0
                 else (path["high"] >= target).any())
        stopped = ((path["high"] >= stop).any() if direction > 0
                  else (path["low"] <= stop).any())
        if filled or stopped:
            continue   # this gap already resolved -- keep looking further back
        if lookback > TIME_STOP_DAYS:
            continue

        gap_date = daily.index[i].date()
        days_elapsed = idx - i
        days_left = TIME_STOP_DAYS - days_elapsed
        time_stop_date = mcal.next_trading_day(gap_date)
        for _ in range(TIME_STOP_DAYS - 1):
            time_stop_date = mcal.next_trading_day(time_stop_date)

        spot, spot_est = None, False
        if for_date is None:
            spot = loader.live_spot()
        if not (spot and np.isfinite(spot) and spot > 0):
            spot, spot_est = float(daily["close"].iloc[idx]), (for_date is None)

        return GapSignal(
            qualified=True, asof=asof, day_note=day_note,
            gap_date=str(gap_date), gap_type=("up" if direction > 0 else "down"),
            spot=spot, spot_estimated=spot_est, gap_open=gap_open,
            target=target, stop=stop, days_elapsed=days_elapsed,
            days_left=days_left, time_stop_date=str(time_stop_date),
            historical_fill_rate=_fill_rate_for(gap_pct, direction),
        )

    return GapSignal(qualified=False, asof=asof, day_note=day_note,
                     setup="No qualifying gap (>=0.15%) is currently open "
                           "within the last 14 trading days.")
