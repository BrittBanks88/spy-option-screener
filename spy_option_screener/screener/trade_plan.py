"""Turn a pullback-before-continuation setup into a full, Slack-ready trade plan.

Focus: SPY is in an established trend (20d > 50d > 100d, or the reverse), has
pulled back, and the ``pullback`` signal has fired -- i.e. it's turning back in
the trend direction. For the best weekly contract we spell out:

    entry (option limit + the SPY level/time that triggers it)
    stop  (-50% of premium, and the SPY level that corresponds)
    target 1 (take-profit, ~+1.3 ATR) and a runner (prior swing)
    time stop
    risk, reward:risk, greeks, breakeven

Prices come from a reconstructed vol surface -- treat as +/-10-20%.
Not investment advice.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..data import intraday as it
from ..data import loader
from ..data import market_calendar as mcal
from ..pricing import black_scholes as bs
from ..signals import indicators as ind
from ..signals import strategies as dsig
from . import chain as chain_mod
from . import score as score_mod
from . import overnight as overnight_mod

ET = ZoneInfo("America/New_York")
R, Q = 0.04, 0.013


def _fmt_d(iso: str) -> str:
    return dt.date.fromisoformat(iso).strftime("%b %d")


@dataclass
class TradePlan:
    qualified: bool
    asof: str
    session_date: str
    live: bool
    day_note: str
    setup: str
    direction: int = 0
    spot: float = 0.0
    atr: float = 0.0
    trend_note: str = ""
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0
    # contract
    kind: str = ""
    strike: float = 0.0
    expiry: str = ""
    dte: int = 0
    delta: float = 0.0
    iv: float = 0.0
    theta: float = 0.0
    breakeven: float = 0.0
    pop: float = 0.0
    score: float = 0.0
    # underlying levels
    u_entry: float = 0.0
    u_entry_estimated: bool = False
    u_stop: float = 0.0
    u_t1: float = 0.0
    u_run: float = 0.0
    entry_hold_level: float = 0.0
    entry_window: str = ""
    entry_quality: str = ""       # "" | "extended — chase small or wait for a dip"
    # option-premium levels
    o_entry: float = 0.0
    o_entry_lo: float = 0.0
    o_entry_hi: float = 0.0
    o_stop: float = 0.0
    o_t1: float = 0.0
    o_run: float = 0.0
    o_stop_pct: float = -0.5
    o_t1_pct: float = 0.0
    o_run_pct: float = 0.0
    # risk
    max_loss: float = 0.0
    reward_risk: float = 0.0
    time_stop: str = ""
    manage_date: str = ""         # "Wed Aug 26" or "Thu Aug 27 (same day)"
    overnight: str = ""           # overnight-carry assessment line
    data_source: str = "reconstructed"   # "polygon" | "reconstructed"
    caveat: str = ("Reconstructed vol surface (±10–20%). The pullback signal is the "
                   "best-behaved in backtests but still only ~breakeven on a ~2-yr "
                   "sample. Size small. Not investment advice.")

    # ----- Slack renderers ----------------------------------------------
    def slack_text(self) -> str:
        if not self.qualified:
            return (f"*SPY pullback screen — {self.session_date}*\n"
                    f"_{self.day_note}_\n\n"
                    f":no_entry: *No pullback-before-continuation setup.* {self.setup}")
        s = "CALL" if self.direction > 0 else "PUT"
        arrow = "above" if self.direction > 0 else "below"
        turn = "turning up" if self.direction > 0 else "rolling over"
        src = {"schwab": "📡 live Schwab chain",
               "polygon": "📡 live Polygon chain",
               "schwab+livespot": "📡 Schwab Greeks + live SPY spot",
               "polygon+livespot": "📡 Polygon Greeks + live SPY spot",
               }.get(self.data_source, "🔒 reconstructed chain")
        tag = f"🟢 live · {src}" if self.live else f"{src} — confirm at the open"
        est = " _(est.)_" if self.u_entry_estimated else ""
        return "\n".join([
            f"*SPY Pullback → Continuation  |  {s}*   _{tag}_",
            f"_{self.day_note}_",
            "",
            f"*Contract:*  SPY ${self.strike:.0f} {s}   ·   exp {self.expiry} "
            f"({self.dte} DTE)   ·   Δ {self.delta:+.2f}   ·   IV {self.iv:.0%}",
            f"*Entry:*  ${self.o_entry:.2f} limit  (fill ${self.o_entry_lo:.2f}–"
            f"${self.o_entry_hi:.2f})",
            f"       ↳ trigger: SPY green on the day and {arrow} your stop ${self.entry_hold_level:.2f} "
            f"at entry time (≈ ${self.u_entry:.2f}{est}); if not, skip it — {self.entry_window}",
            f"*Stop:*  −50% premium → ${self.o_stop:.2f}   "
            f"(SPY {'<' if self.direction > 0 else '>'} ${self.u_stop:.2f})",
            f"*Target 1:*  ${self.o_t1:.2f}  ({self.o_t1_pct:+.0%})   at SPY "
            f"${self.u_t1:.2f}   → take ⅔, trail the rest",
            f"*Runner:*  ${self.o_run:.2f}  ({self.o_run_pct:+.0%})   at the prior "
            f"swing ${self.u_run:.2f}",
            f"*Time stop:*  {self.time_stop}",
            *( [self.overnight] if self.overnight else [] ),
            *( [f"*Heads-up:*  {self.entry_quality}"] if self.entry_quality else [] ),
            "",
            f"*Risk:*  ${self.max_loss:.0f}/contract max   ·   reward:risk ≈ "
            f"{self.reward_risk:.1f} : 1 to T1   ·   breakeven ${self.breakeven:.2f}"
            f"   ·   POP {self.pop:.0%}   ·   score {self.score:.0f}/100",
            f"*Why:*  {' · '.join(self.reasons)}   (conviction {self.confidence:.0%})",
            "",
            f"_{self.caveat}_",
        ])

    def slack_blocks(self) -> list[dict]:
        if not self.qualified:
            return [
                {"type": "header", "text": {"type": "plain_text",
                 "text": f"SPY pullback screen — {self.session_date}"}},
                {"type": "section", "text": {"type": "mrkdwn",
                 "text": f":no_entry: *No pullback-before-continuation setup.*\n"
                         f"{self.setup}\n_{self.day_note}_"}},
            ]
        s = "CALL" if self.direction > 0 else "PUT"
        cmp = "<" if self.direction > 0 else ">"
        fields = [
            ("Contract", f"SPY ${self.strike:.0f} {s}\nexp {self.expiry} · {self.dte} DTE\nΔ {self.delta:+.2f} · IV {self.iv:.0%}"),
            ("Entry", f"${self.o_entry:.2f} limit\n(${self.o_entry_lo:.2f}–${self.o_entry_hi:.2f})\nSPY ≈ ${self.u_entry:.2f}"),
            ("Stop", f"−50% → ${self.o_stop:.2f}\nSPY {cmp} ${self.u_stop:.2f}"),
            ("Target 1", f"${self.o_t1:.2f} ({self.o_t1_pct:+.0%})\nSPY ${self.u_t1:.2f}\ntake ⅔"),
            ("Runner", f"${self.o_run:.2f} ({self.o_run_pct:+.0%})\nSPY ${self.u_run:.2f}"),
            ("Risk / RR", f"${self.max_loss:.0f}/ct max\n≈ {self.reward_risk:.1f}:1 to T1\nPOP {self.pop:.0%}"),
        ]
        return [
            {"type": "header", "text": {"type": "plain_text",
             "text": f"SPY Pullback → Continuation · {s}"}},
            {"type": "context", "elements": [{"type": "mrkdwn",
             "text": f"{'🟢 live' if self.live else '🔒 reconstructed'} · {self.day_note}"}]},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*{k}*\n{v}"} for k, v in fields]},
            {"type": "section", "text": {"type": "mrkdwn", "text": (
                f"*Entry trigger:* SPY green on the day, "
                f"{'above' if self.direction > 0 else 'below'} the stop "
                f"${self.entry_hold_level:.2f} — {self.entry_window}"
                + (f"\n:warning: {self.entry_quality}" if self.entry_quality else "")
                + f"\n*Time stop:* {self.time_stop}"
                + (f"\n{self.overnight}" if self.overnight else "")
                + "\n"
                f"*Why:* {' · '.join(self.reasons)} (conviction {self.confidence:.0%})\n"
                f"breakeven ${self.breakeven:.2f} · score {self.score:.0f}/100")}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": self.caveat}]},
        ]

    # ----- simple / "just tell me what to do" renderers ------------------
    def _why_short(self) -> str:
        return ("dip in an uptrend, bouncing" if self.direction > 0
                else "bounce in a downtrend, rolling over")

    def _simple_body(self) -> str:
        cmp = "above" if self.direction > 0 else "below"
        when = "now" if self.live else "around 10am ET"
        lines = [
            f"*GO IF* → SPY is green and {cmp} ${self.entry_hold_level:.0f} {when}",
            f"*BUY* → about ${self.o_entry:.2f}   (${self.o_entry * 100:.0f} per contract)",
            f"*STOP* → ${self.o_stop:.2f}   ·  sell here if it drops, no exceptions",
            f"*TARGET* → ${self.o_t1:.2f}  (+{self.o_t1_pct:.0%})   ·  sell most of it here",
            f"*DONE BY* → {self.manage_date}   ·  don't hold to expiry",
        ]
        if self.entry_quality:
            lines.append("*NOTE* → it already ran up — smaller size, or wait for a dip")
        return "\n".join(lines)

    def slack_simple_text(self) -> str:
        if not self.qualified:
            return ("😴 *No SPY setup today.*  Nothing to do — "
                    "you'll get a ping when there's a real one.")
        s = "CALL" if self.direction > 0 else "PUT"
        return (
            f"📈 *SPY ${self.strike:.0f} {s}*  ·  exp {_fmt_d(self.expiry)} "
            f"({self.dte} days left)\n\n"
            + self._simple_body()
            + f"\n\n_why: {self._why_short()} · payoff about "
            f"{self.reward_risk:.1f} to 1_\n"
            + ("_📡 live price · not financial advice_"
               if self.data_source.startswith("polygon")
               else "_🔒 reconstructed price · not financial advice_")
        )

    def slack_simple_blocks(self) -> list[dict]:
        if not self.qualified:
            return [{"type": "section", "text": {"type": "mrkdwn",
                     "text": "😴 *No SPY setup today.*  Nothing to do."}}]
        s = "CALL" if self.direction > 0 else "PUT"
        return [
            {"type": "header", "text": {"type": "plain_text",
             "text": f"SPY ${self.strike:.0f} {s} — buy setup"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": self._simple_body()}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": (
                f"exp {_fmt_d(self.expiry)} ({self.dte}d) · why: {self._why_short()} · "
                f"payoff ~{self.reward_risk:.1f}:1 · reconstructed price, not advice")}]},
        ]

    def slack_payload(self, style: str = "full") -> dict:
        """Incoming-webhook body. style='simple' for the ADHD-friendly version."""
        if style == "simple":
            fallback = self.slack_simple_text().split("\n")[0].replace("*", "")
            return {"text": fallback, "blocks": self.slack_simple_blocks()}
        first = self.slack_text().split("\n\n")[0].replace("*", "").replace("_", "")
        return {"text": first, "blocks": self.slack_blocks()}


# ---------------------------------------------------------------------------
def _market_ctx(session_date: dt.date):
    now = dt.datetime.now(ET)
    today = now.date()
    is_sess = mcal.is_trading_day(today)
    live = (is_sess and dt.time(9, 30) <= now.time() < dt.time(16, 0)
            and session_date == today)
    nxt = today if (is_sess and today > session_date) \
        else mcal.next_trading_day(max(today, session_date))
    if live:
        note = f"Live · {today:%a %b %d}."
    elif not is_sess:
        why = "weekend" if today.weekday() >= 5 else "holiday"
        note = f"{today:%a %b %d} — market closed ({why}). Next session {nxt:%a %b %d}."
    else:
        note = f"{today:%a %b %d}. Next session {nxt:%a %b %d}."
    return live, nxt, note


def _t_years(dte: int, hold_days: int) -> float:
    return max(dte - hold_days, 0.5) / 365.0


def build_plan(entry_time="10:00", interval="1h", buy_zone=(0.35, 0.60),
               refresh=False, hist_start="2016-01-01",
               for_date: str | dt.date | None = None) -> TradePlan:
    """Plan for the latest session, or for ``for_date`` (must have intraday bars)."""
    daily = loader.load_history(start=hist_start, refresh=refresh)
    daily.index = pd.DatetimeIndex(daily.index)
    bars = it.load_intraday(interval, refresh=refresh, live_today=(for_date is None))
    session_date = (pd.Timestamp(for_date).date() if for_date is not None
                    else it.latest_session(interval, live_today=True)[0])
    live, _next, wall_note = _market_ctx(session_date)
    asof = dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M %Z")

    entry_date = mcal.next_trading_day(session_date)
    day_note = (f"Signal: {session_date:%a %b %d} close  →  enter {entry_date:%a %b %d}"
                if for_date is not None else wall_note)
    dsig_daily = daily.loc[:pd.Timestamp(session_date)]

    # --- qualify the setup ---------------------------------------------
    pb_row = dsig.pullback(dsig_daily).iloc[-1]
    pdir = int(pb_row["direction"])
    close = dsig_daily["close"]
    sma20, sma50, sma100 = (ind.sma(close, 20).iloc[-1], ind.sma(close, 50).iloc[-1],
                            ind.sma(close, 100).iloc[-1])
    up_stack, dn_stack = sma20 > sma50 > sma100, sma20 < sma50 < sma100
    trend_ok = (pdir > 0 and up_stack) or (pdir < 0 and dn_stack)
    trend_note = ("uptrend: 20d > 50d > 100d" if up_stack else
                  "downtrend: 20d < 50d < 100d" if dn_stack else "no clean MA trend")

    base = TradePlan(qualified=False, asof=asof, session_date=str(session_date),
                     live=live, day_note=day_note, setup="")
    if pdir == 0:
        base.setup = f"No pullback firing. {trend_note}."
        return base
    if not trend_ok:
        base.setup = (f"Pullback fired ({'calls' if pdir > 0 else 'puts'}) but the MA "
                      f"trend doesn't confirm continuation — {trend_note}.")
        return base

    # --- entry price ---------------------------------------------------
    or_min = 30 if interval != "1h" else 60
    feat = it.daily_features(bars, daily, or_minutes=or_min, signal_cutoff=entry_time,
                             entry_times=(entry_time, "10:00", "10:30", "11:00"))
    px_col = f"px_{entry_time.replace(':', '')}"
    erows = feat[feat.index.date == entry_date]
    if len(erows) and np.isfinite(erows.iloc[0].get(px_col, np.nan)):
        u_entry, u_entry_est = float(erows.iloc[0][px_col]), False
    else:
        u_entry, u_entry_est = float(close.iloc[-1]), True

    vix = float(dsig_daily["vix"].iloc[-1])
    rv = float(dsig_daily["realized_vol_20"].iloc[-1])
    atr = float(ind.atr(dsig_daily["high"], dsig_daily["low"], close, 14).iloc[-1])
    sig_close = float(close.iloc[-1])
    _ext = (u_entry - sig_close) * pdir            # move since the signal, in dir
    entry_quality = ("" if _ext <= 0.6 * atr else
                     f"entry is {_ext/atr:.1f} ATR beyond the signal close "
                     f"(${sig_close:.2f}) — chase small, or wait for a dip toward "
                     f"the 10-day line")

    # --- contract -----------------------------------------------------
    exp_date, dte = mcal.next_weekly_expiry(entry_date, min_dte=3, max_dte=9)
    chain = chain_mod.get_chain(
        u_entry, vix, dte, vix_baseline=float(dsig_daily["vix"].tail(40).mean()),
        min_dte=3, max_dte=9,
        prefer_live=(for_date is None))     # back-checks stay on the reconstruction
    chain_source = chain.attrs.get("source", "reconstructed")
    live_chain = chain_source in ("polygon", "schwab")
    reanchored = bool(chain.attrs.get("reanchored"))
    if live_chain and len(chain):
        exp_date = pd.Timestamp(chain["expiry"].iloc[0]).date()
        dte = int(chain["dte"].iloc[0])
        u_entry = float(chain.attrs.get("spot", u_entry))
    scored = score_mod.score_chain(chain, pdir, u_entry, rv, confidence=1.0)
    lo, hi = buy_zone
    band = scored[scored["delta"].abs().between(lo, hi)]
    c = (band if not band.empty else scored).iloc[0]
    strike, kind, iv = float(c["strike"]), c["type"], float(c["iv"])
    o_entry = float(c["mid"])

    # --- underlying levels ------------------------------------------
    ma10 = float(ind.sma(close, 10).iloc[-1])
    buf = 0.0009 * u_entry
    if pdir > 0:
        pull_low = float(dsig_daily["low"].tail(3).min())
        u_stop = max(min(pull_low - buf, u_entry - 0.8 * atr), u_entry - 1.3 * atr)
        u_t1 = u_entry + 1.0 * atr
        swing_hi = float(dsig_daily["high"].tail(20).max())
        u_run = min(max(swing_hi, u_entry + 1.8 * atr), u_entry + 2.8 * atr)
        hold_level = u_stop
    else:
        pull_hi = float(dsig_daily["high"].tail(3).max())
        u_stop = min(max(pull_hi + buf, u_entry + 0.8 * atr), u_entry + 1.3 * atr)
        u_t1 = u_entry - 1.0 * atr
        swing_lo = float(dsig_daily["low"].tail(20).min())
        u_run = max(min(swing_lo, u_entry - 1.8 * atr), u_entry - 2.8 * atr)
        hold_level = u_stop

    # --- option-premium levels (priced ~1.5 days out) -----------------
    t_red = _t_years(dte, min(2, max(1, dte - 1)))
    o_stop = 0.50 * o_entry                                  # stated stop
    o_t1 = max(float(bs.price(u_t1, strike, t_red, iv * 0.93, R, Q, kind)),
               1.35 * o_entry)
    o_run = max(float(bs.price(u_run, strike, t_red, iv * 0.88, R, Q, kind)),
                1.8 * o_entry)
    o_t1_pct, o_run_pct = o_t1 / o_entry - 1.0, o_run / o_entry - 1.0
    reward_risk = (o_t1 - o_entry) / (o_entry - o_stop)

    manage = exp_date - dt.timedelta(days=1)
    while not mcal.is_trading_day(manage):
        manage -= dt.timedelta(days=1)
    same_day = manage <= entry_date
    manage_date = (f"{exp_date:%a %b %d} (same day — manage intraday)" if same_day
                   else f"{manage:%a %b %d}")
    time_stop = (f"manage intraday on {exp_date:%a %b %d} — don't hold into the "
                 f"last hour" if same_day else
                 f"close by {manage:%a %b %d} if neither level has hit — "
                 f"don't carry into the {exp_date:%a %b %d} expiration")

    return TradePlan(
        qualified=True, asof=asof, session_date=str(session_date), live=live,
        day_note=day_note,
        setup="Pullback inside an established trend — playing the continuation.",
        direction=pdir, spot=u_entry, atr=atr, trend_note=trend_note,
        reasons=[f"{pb_row['edge_note']} ({pb_row['confidence']:.0%})", trend_note,
                 f"daily ATR ≈ ${atr:.2f}"],
        confidence=float(pb_row["confidence"]),
        kind=kind, strike=strike, expiry=str(exp_date), dte=dte,
        delta=float(c["delta"]), iv=iv, theta=float(c["theta"]),
        breakeven=float(c["breakeven"]), pop=float(c["chance_of_profit"]),
        score=float(c["score"]),
        u_entry=u_entry, u_entry_estimated=(u_entry_est and not live_chain),
        u_stop=u_stop, u_t1=u_t1, u_run=u_run, entry_hold_level=hold_level,
        entry_quality=entry_quality,
        entry_window=("enter now if it's holding the level" if live
                      else f"~{entry_time}–10:30 ET on {entry_date:%a %b %d}"),
        o_entry=o_entry, o_entry_lo=o_entry * 0.97, o_entry_hi=o_entry * 1.04,
        o_stop=o_stop, o_t1=o_t1, o_run=o_run,
        o_stop_pct=-0.5, o_t1_pct=o_t1_pct, o_run_pct=o_run_pct,
        max_loss=o_entry * 100.0, reward_risk=reward_risk, time_stop=time_stop,
        manage_date=manage_date,
        overnight=overnight_mod.assess(dsig_daily, session_date, pdir).line(pdir),
        data_source=(f"{chain_source}+livespot" if (live_chain and reanchored)
                     else chain_source if live_chain else "reconstructed"),
        caveat=(
            f"{chain_source.title()} Greeks re-priced to a live SPY quote (the "
            "chain quote was stale). Pullback is the best-behaved signal in "
            "backtests, still only ~breakeven on a ~2-yr sample. Size small. "
            "Not investment advice."
            if (live_chain and reanchored) else
            f"Live {chain_source.title()} option chain. The pullback signal is "
            "the best-behaved in backtests but still only ~breakeven on a ~2-yr "
            "sample. Size small. Not investment advice." if live_chain else
            "Reconstructed vol surface (±10–20%). The pullback signal is the "
            "best-behaved in backtests but still only ~breakeven on a ~2-yr "
            "sample. Size small. Not investment advice."),
    )
