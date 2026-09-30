"""Read-only price-target view: where SPY might go, by when, and which side
that favors — no strike, no entry price, no stop, no position mechanics.

Uses the SAME two signals validated in screener/trade_plan.py, one per
direction, instead of an equal-weight blend of all five daily signals (most
of which have no established edge for SPY specifically):

    bullish -> pullback inside a confirmed uptrend   (2024-2026: 55% win,
               PF 1.5 backtested on the weekly trade plan)
    bearish -> momentum breakdown below the recent range + below the 100d
               trend (worked independently in 2018 AND 2022: 59-60% win,
               PF 3-5.5 -- but lost money in 2024-2026's uptrend, so a
               bearish read fired before the broader trend actually turns
               carries a regime_warning)

Targets/timing are still ATR-scaled heuristics layered on top of a validated
directional signal -- see research/backtest_price_target.py for the honest,
separately-measured hit-rate of the levels themselves (not just the direction
call). Informational only, not a trade recommendation.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data import intraday as it
from ..data import loader
from ..data import market_calendar as mcal
from ..signals import indicators as ind
from ..signals import strategies as dsig
from .trade_plan import _market_ctx   # reuse the day-awareness helper

MIN_CONFIDENCE = 0.30


def _blend(dsig_daily: pd.DataFrame) -> tuple[int, float, list[str], str]:
    """Direction + confidence from the two validated signals only. Returns
    (direction, confidence, reasons, regime_warning)."""
    close = dsig_daily["close"]
    sma20, sma50, sma100 = (ind.sma(close, 20).iloc[-1], ind.sma(close, 50).iloc[-1],
                            ind.sma(close, 100).iloc[-1])
    up_stack, dn_stack = sma20 > sma50 > sma100, sma20 < sma50 < sma100

    pb = dsig.pullback(dsig_daily).iloc[-1]
    mb = dsig.momentum_breakout(dsig_daily).iloc[-1]

    if int(pb["direction"]) > 0 and up_stack:
        return (1, float(pb["confidence"]),
                [f"pullback in a confirmed uptrend ({pb['confidence']:.0%}) — "
                 f"{pb['edge_note']}"], "")
    if int(mb["direction"]) < 0:
        warning = ("" if dn_stack else
                  "⚠️ Broader trend isn't confirmed bearish yet — this exact "
                  "bearish read lost money in similar 2024–2026 conditions "
                  "(33% win, −6% avg). Its real edge (59–60% win) only showed "
                  "up during confirmed bear markets (2018, 2022).")
        return (-1, float(mb["confidence"]),
                [f"momentum breakdown ({mb['confidence']:.0%}) — {mb['edge_note']}"],
                warning)
    return 0, 0.0, [], ""


@dataclass
class PriceTargetView:
    qualified: bool
    asof: str
    session_date: str
    live: bool
    day_note: str
    setup: str = ""
    direction: int = 0
    confidence: float = 0.0
    spot: float = 0.0
    spot_estimated: bool = False
    atr: float = 0.0
    near_target: float = 0.0
    near_window: str = "1–2 sessions"
    far_target: float = 0.0
    far_window: str = "within the week (~5 sessions)"
    watch_level: float = 0.0
    reasons: list[str] = field(default_factory=list)
    regime_warning: str = ""
    caveat: str = ("Reconstructed price. Backtested hit-rate (2018-2026, n=149): "
                   "near target reached first 55% of the time (median 2 sessions "
                   "when it hits); far target 35% (median 4 sessions); the watch/"
                   "cut level got hit FIRST 44% of the time — almost a coin flip. "
                   "See research/backtest_price_target.py. Informational only.")

    @property
    def bias(self) -> str:
        return "Bullish" if self.direction > 0 else "Bearish" if self.direction < 0 else "Flat"

    @property
    def favors(self) -> str:
        return ("favors calls over puts" if self.direction > 0 else
                "favors puts over calls" if self.direction < 0 else
                "no lean either way")

    # ----- Slack renderers ----------------------------------------------
    def slack_text(self) -> str:
        if not self.qualified:
            return (f"📊 *SPY price view — {self.session_date}*\n"
                    f"_{self.day_note}_\n\n"
                    f"No clear directional lean right now. {self.setup}")
        est = " _(est.)_" if self.spot_estimated else ""
        near_pct = self.near_target / self.spot - 1.0
        far_pct = self.far_target / self.spot - 1.0
        watch_pct = self.watch_level / self.spot - 1.0
        tag = "🟢 live" if self.live else "🔒 reconstructed"
        return "\n".join([
            f"📊 *SPY Price View*  ·  spot ${self.spot:.2f}{est}   _{tag}_",
            f"_{self.day_note}_",
            "",
            f"*Bias:* {self.bias} (conviction {self.confidence:.0%}) — "
            f"{self.favors}",
            f"*Near target:* ${self.near_target:.2f}  ({near_pct:+.1%})   "
            f"·  typically {self.near_window}",
            f"*Extended target:* ${self.far_target:.2f}  ({far_pct:+.1%})   "
            f"·  {self.far_window}",
            f"*Watch level:* ${self.watch_level:.2f}  ({watch_pct:+.1%})   "
            f"·  a close beyond here would weaken this read",
            *( [self.regime_warning] if self.regime_warning else [] ),
            "",
            f"*Why:* {' · '.join(self.reasons)}",
            "",
            f"_{self.caveat}_",
        ])

    def slack_blocks(self) -> list[dict]:
        if not self.qualified:
            return [
                {"type": "header", "text": {"type": "plain_text",
                 "text": f"SPY price view — {self.session_date}"}},
                {"type": "section", "text": {"type": "mrkdwn",
                 "text": f"No clear directional lean right now. {self.setup}\n"
                         f"_{self.day_note}_"}},
            ]
        fields = [
            ("Bias", f"{self.bias} ({self.confidence:.0%})\n{self.favors}"),
            ("Near target", f"${self.near_target:.2f}\n{self.near_window}"),
            ("Extended target", f"${self.far_target:.2f}\n{self.far_window}"),
            ("Watch level", f"${self.watch_level:.2f}\ninvalidates the read"),
        ]
        return [
            {"type": "header", "text": {"type": "plain_text",
             "text": f"SPY Price View · {self.bias}"}},
            {"type": "context", "elements": [{"type": "mrkdwn",
             "text": f"{'🟢 live' if self.live else '🔒 reconstructed'} · "
                     f"spot ${self.spot:.2f} · {self.day_note}"}]},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*{k}*\n{v}"} for k, v in fields]},
            {"type": "section", "text": {"type": "mrkdwn",
             "text": (f"{self.regime_warning}\n" if self.regime_warning else "")
             + f"*Why:* {' · '.join(self.reasons)}"}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": self.caveat}]},
        ]

    def slack_payload(self) -> dict:
        first = self.slack_text().split("\n\n")[0].replace("*", "")
        return {"text": first, "blocks": self.slack_blocks()}


def build_view(interval: str = "1h", hist_start: str = "2016-01-01",
              refresh: bool = False,
              for_date: str | dt.date | None = None) -> PriceTargetView:
    daily = loader.load_history(start=hist_start, refresh=refresh)
    daily.index = pd.DatetimeIndex(daily.index)
    bars = it.load_intraday(interval, refresh=refresh, live_today=(for_date is None))

    session_date = (pd.Timestamp(for_date).date() if for_date is not None
                    else it.latest_session(interval, live_today=True)[0])
    live, _next, wall_note = _market_ctx(session_date)
    asof = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    day_note = (f"As of {session_date:%a %b %d} close" if for_date is not None
                else wall_note)

    dsig_daily = daily.loc[:pd.Timestamp(session_date)] if for_date else daily
    last_daily = dsig_daily.index[-1]
    close = dsig_daily["close"]

    direction, confidence, reasons, regime_warning = _blend(dsig_daily)

    base = PriceTargetView(qualified=False, asof=asof, session_date=str(session_date),
                           live=live, day_note=day_note)
    if direction == 0 or confidence < MIN_CONFIDENCE:
        base.setup = (f"Neither signal is firing (confidence {confidence:.0%}, "
                      f"need ≥{MIN_CONFIDENCE:.0%}) as of {last_daily.date():%a %b %d}.")
        return base

    # spot: for a live read, near-real-time if we can get it, else the last
    # daily close (flagged as an estimate). For a --for back-check, ALWAYS use
    # that date's actual close -- never today's live quote.
    spot, spot_est = None, False
    if for_date is None:
        spot = loader.live_spot()
    if not (spot and np.isfinite(spot) and spot > 0):
        spot, spot_est = float(close.iloc[-1]), (for_date is None)

    atr = float(ind.atr(dsig_daily["high"], dsig_daily["low"], close, 14).iloc[-1])
    near = spot + direction * 1.0 * atr
    if direction > 0:
        swing = float(dsig_daily["high"].tail(20).max())
        far = min(max(swing, spot + 1.8 * atr), spot + 2.8 * atr)
    else:
        swing = float(dsig_daily["low"].tail(20).min())
        far = max(min(swing, spot - 1.8 * atr), spot - 2.8 * atr)
    watch = spot - direction * 1.0 * atr

    base.qualified = True
    base.direction, base.confidence = direction, confidence
    base.spot, base.spot_estimated = spot, spot_est
    base.atr = atr
    base.near_target, base.far_target, base.watch_level = near, far, watch
    base.reasons = reasons
    base.regime_warning = regime_warning
    return base
