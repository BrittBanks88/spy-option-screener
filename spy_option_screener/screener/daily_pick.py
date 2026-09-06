"""'Best contract to buy today' — combine the daily + intraday signals, then
rank the nearest-weekly chain and return the single best long call/put.

Price is NOT a filter: the pick is chosen purely on the screener score (and,
by default, restricted to the sane directional-buy delta band, not by cost).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

from ..data import intraday as it
from ..data import loader
from ..data import market_calendar as mcal
from ..pricing import vol_model as vm
from ..signals import intraday_signals as isig
from ..signals import strategies as dsig
from . import chain as chain_mod
from . import score as score_mod


@dataclass
class DailyPick:
    asof: str
    session_date: str
    market_state: str            # "live" | "after close" | "closed"
    spot: float
    vix: float
    realized_vol: float
    gap: float
    direction: int               # +1 call, -1 put, 0 none
    conviction: float            # 0..1
    reasons: list[str] = field(default_factory=list)
    expiry: str = ""
    dte: int = 0
    contract: dict | None = None      # the winning row
    call_alt: dict | None = None      # best call regardless of direction
    put_alt: dict | None = None       # best put regardless of direction
    note: str = ""
    today: str = ""                   # calendar date this was run
    today_weekday: str = ""
    next_session: str = ""            # next date the market is open
    next_session_weekday: str = ""
    live: bool = False               # signals reflect a session in progress
    day_note: str = ""               # plain-English "where we are" line

    def headline(self) -> str:
        tense = "" if self.live else " (from the last completed session)"
        if self.direction == 0 or self.contract is None:
            return (f"No strong directional setup{tense}. "
                    f"Gap {self.gap:+.2%}, conviction {self.conviction:.0%}. "
                    "Stand aside; see call/put alternates below.")
        c = self.contract
        side = "CALL" if self.direction > 0 else "PUT"
        verb = "BUY" if self.live else "SETUP:"
        return (f"{verb}  SPY ${c['strike']:.0f} {side}  exp {self.expiry} ({self.dte}d)"
                f"  ·  ~${c['mid']:.2f} (${c['mid']*100:.0f}/contract)"
                f"  ·  breakeven ${c['breakeven']:.2f}"
                f"  ·  chance of profit {c['chance_of_profit']:.0%}"
                f"  ·  score {c['score']:.0f}/100{tense}")


def _market_context(session_date: dt.date):
    """Return (state, live, next_session, day_note) given the wall clock (ET)."""
    now = dt.datetime.now(ET)
    today = now.date()
    open_t, close_t = dt.time(9, 30), dt.time(16, 0)
    today_is_session = mcal.is_trading_day(today)

    if today_is_session and open_t <= now.time() < close_t:
        state, live = "live", session_date == today
    elif today_is_session and now.time() >= close_t and session_date == today:
        state, live = "after close", False
    else:
        state, live = "closed", False

    if today_is_session and now.time() < open_t:
        nxt = today                       # market opens later today
    elif today_is_session and now.time() < close_t:
        nxt = today                       # in session now
    else:
        nxt = mcal.next_trading_day(today)

    bits = [f"Today is {today:%A %b %d}."]
    if state == "live":
        bits.append("Market is open — this is a live read.")
    elif state == "after close":
        bits.append("Market has closed for the day.")
    else:
        if not today_is_session:
            why = "weekend" if today.weekday() >= 5 else "market holiday"
            bits.append(f"Market is closed ({why}).")
        else:
            bits.append("Market has not opened yet.")
    if nxt != today or state != "live":
        # note any holiday gap between now and the next session
        gap_days = [today + dt.timedelta(d) for d in range(1, (nxt - today).days)]
        hols = [d for d in gap_days if d.weekday() < 5 and not mcal.is_trading_day(d)]
        extra = (f" ({', '.join(f'{h:%A}' for h in hols)} is a holiday)"
                 if hols else "")
        bits.append(f"Next session: {nxt:%A %b %d}{extra}.")
    if session_date != nxt and state != "live":
        bits.append(f"The signals below reconstruct the last completed session "
                    f"({session_date:%a %b %d}); re-run after ~10:30 ET on "
                    f"{nxt:%b %d} for a live pick.")
    return state, live, nxt, " ".join(bits)


def _row(scored: pd.DataFrame) -> dict | None:
    if scored is None or scored.empty:
        return None
    r = scored.iloc[0]
    keep = ["type", "strike", "dte", "mid", "iv", "delta", "gamma", "theta",
            "vega", "breakeven", "be_pct", "chance_of_profit", "max_loss",
            "leverage", "exp_move", "score", "s_breakeven", "s_delta",
            "s_theta", "s_vol_value", "s_liquidity"]
    return {k: (float(r[k]) if k != "type" else r[k]) for k in keep if k in r}


def todays_pick(interval="30m", entry_time="10:30", buy_zone=True,
                refresh=False, hist_start="2018-01-01") -> DailyPick:
    daily = loader.load_history(start=hist_start, refresh=refresh)
    daily.index = pd.DatetimeIndex(daily.index)

    bars = it.load_intraday(interval, refresh=refresh)
    session_date, day_bars = it.latest_session(interval, refresh=False)
    state, live, next_session, day_note = _market_context(session_date)

    or_minutes = 30 if interval != "1h" else 60
    feat_all = it.daily_features(bars, daily, or_minutes=or_minutes,
                                 signal_cutoff=entry_time,
                                 entry_times=(entry_time, "10:00", "10:30", "11:00"))
    if session_date not in [d.date() for d in feat_all.index]:
        raise RuntimeError("no features for the latest session")
    feat = feat_all[feat_all.index.date == session_date]

    # signals ------------------------------------------------------------
    db = dsig.combine(daily)
    db.index = pd.DatetimeIndex(db.index)
    sig = isig.combine_intraday(feat, daily_bias=db).iloc[0]
    gap_row = isig.gap_continuation(feat).iloc[0]
    orb_row = isig.opening_range_breakout(feat).iloc[0]

    daily_rows = [("Gap", gap_row), ("Opening range", orb_row)]
    for sname in ("pullback", "trend_continuation", "mean_reversion",
                  "momentum_breakout", "vol_regime"):
        srow = dsig.STRATEGIES[sname](daily)
        srow.index = pd.DatetimeIndex(srow.index)
        r = srow.reindex(feat.index).iloc[0]
        daily_rows.append((sname.replace("_", " ").title(), r))

    reasons = []
    for label, r in daily_rows:
        if int(r["direction"]) != 0:
            arrow = "calls" if r["direction"] > 0 else "puts"
            reasons.append(f"{label}: {arrow} ({r['confidence']:.0%}) — {r['edge_note']}")
        else:
            reasons.append(f"{label}: neutral")

    direction = int(sig["direction"])
    conviction = float(sig["confidence"])

    # chain ------------------------------------------------------------
    frow = feat.iloc[0]
    spot = float(frow.get(f"px_{entry_time.replace(':', '')}", frow["close"]))
    if not np.isfinite(spot) or spot <= 0:
        spot = float(frow["close"])
    vix = float(frow["vix"]) if np.isfinite(frow["vix"]) else float(daily["vix"].iloc[-1])
    rv = float(frow["realized_vol_20"]) if np.isfinite(frow["realized_vol_20"]) \
        else float(daily["realized_vol_20"].iloc[-1])
    vix_base = float(daily["vix"].tail(40).mean())

    exp_date, dte = mcal.next_weekly_expiry(session_date)
    chain = chain_mod.synthetic_chain(spot, vix, dte, vix_baseline=vix_base)

    def best_side(dir_):
        sc = score_mod.score_chain(chain, dir_, spot, rv, confidence=1.0)
        if buy_zone and not sc.empty:
            sc = sc[sc["delta"].abs().between(0.20, 0.62)]
        return _row(sc)

    call_alt, put_alt = best_side(1), best_side(-1)
    contract = None
    if direction > 0:
        contract = call_alt
    elif direction < 0:
        contract = put_alt

    now_et = dt.datetime.now(ET)
    return DailyPick(
        asof=now_et.strftime("%Y-%m-%d %H:%M %Z"),
        session_date=str(session_date),
        market_state=state,
        spot=spot, vix=vix, realized_vol=rv, gap=float(frow["gap"]),
        direction=direction, conviction=conviction, reasons=reasons,
        expiry=str(exp_date), dte=dte, contract=contract,
        call_alt=call_alt, put_alt=put_alt,
        today=str(now_et.date()), today_weekday=now_et.strftime("%A"),
        next_session=str(next_session),
        next_session_weekday=next_session.strftime("%A"),
        live=live, day_note=day_note,
        note=("Intraday history is ~2 years (1h) / ~60 days (30m); the opening-"
              "range signal in particular is lightly tested. Reconstructed "
              "prices ±10-20%. Not investment advice."),
    )
