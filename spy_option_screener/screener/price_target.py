"""Read-only price-target view: where SPY might go, by when, and which side
that favors — no strike, no entry price, no stop, no position mechanics.

Blends all five daily signals (pullback, trend_continuation, mean_reversion,
momentum_breakout, vol_regime) rather than gating on pullback alone, so this
fires more often than the trade plan. Targets/timing are ATR-scaled heuristics,
NOT a separately backtested hit-rate like the pullback trade plan is -- said
plainly in the caveat every time. Informational only, not a trade recommendation.
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
_DAILY_STRATEGIES = ("pullback", "trend_continuation", "mean_reversion",
                    "momentum_breakout", "vol_regime")


def _blend(dsig_daily: pd.DataFrame) -> tuple[int, float, list[str]]:
    """Direction + confidence from whichever daily signals fire, without
    diluting confidence by the full 5-signal universe the way a fixed-weight
    average would (one signal firing at 55% should read as ~55%, not 11%).
    Multiple agreeing signals get a small confidence boost.
    """
    rows = {n: dsig.STRATEGIES[n](dsig_daily).iloc[-1] for n in _DAILY_STRATEGIES}
    score = sum((r["direction"] or 0) * (r["confidence"] or 0) for r in rows.values())
    direction = 0 if score == 0 else (1 if score > 0 else -1)
    reasons = []
    if direction == 0:
        return 0, 0.0, reasons

    agreeing = {n: r for n, r in rows.items() if int(r["direction"] or 0) == direction}
    avg_conf = sum(r["confidence"] for r in agreeing.values()) / len(agreeing)
    confidence = min(1.0, avg_conf * (1 + 0.15 * (len(agreeing) - 1)))
    for n, r in agreeing.items():
        reasons.append(f"{n.replace('_', ' ')} ({r['confidence']:.0%})"
                       + (f" — {r['edge_note']}" if r["edge_note"] else ""))
    return direction, confidence, reasons


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
    caveat: str = ("Reconstructed price. Targets/timing are ATR-scaled estimates, "
                   "not a separately measured hit-rate (unlike the pullback trade "
                   "plan, which is backtested). Informational only — not a trade "
                   "recommendation.")

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
             "text": f"*Why:* {' · '.join(self.reasons)}"}},
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

    direction, confidence, reasons = _blend(dsig_daily)

    base = PriceTargetView(qualified=False, asof=asof, session_date=str(session_date),
                           live=live, day_note=day_note)
    if direction == 0 or confidence < MIN_CONFIDENCE:
        base.setup = (f"Blended signal confidence is {confidence:.0%} "
                      f"(need ≥{MIN_CONFIDENCE:.0%}) as of {last_daily.date():%a %b %d}.")
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
    return base
