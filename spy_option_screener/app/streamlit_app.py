"""SPY gap-fill screener -- signal page + accuracy tracker.

Tab 1: detects an open (unfilled) overnight/opening gap and shows a price
target (the pre-gap level), a stop (2.0x ATR beyond the gap open), and a
14-trading-day time stop. No strike, no premium, no contract mechanics --
see screener/gap_screener.py for the full backtest this is built on (26+
years of SPY daily bars: 85% hit target, 14% stop out, 1% time out).

Tab 2: a read-only accuracy log of contracts tracked against those signals,
stored in a Google Sheet (data/tracker_store.py). Target/stop/time-out are
checked automatically against real SPY price data -- that's the screener's
own accuracy. Entry/exit prices and P&L come from the user's real brokerage
screenshots, added via a one-off script, not reconstructed -- that's their
real trade result. Two different numbers, kept deliberately separate.

    streamlit run streamlit_app.py

The live parts (price, signal, tracker) re-check themselves every minute --
see live_page() -- so an open page never keeps showing an old alert.

Manual only -- nothing on this page places, sizes, or manages a trade.
Not investment advice.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Streamlit re-runs this script after a code change but keeps modules it has
# already imported, so edits to the package (loader, tracker, ...) were ignored
# until the server restarted -- the deployed app once crashed on a stale
# loader. When the package's files change, drop its modules once so the next
# import is current. (Only on change: swapping modules on every run races with
# other open sessions that are mid-import.)
@st.cache_resource
def _code_version() -> dict:
    return {"sig": None}


_sig = max(f.stat().st_mtime_ns for f in (_ROOT / "spy_option_screener").rglob("*.py"))
_ver = _code_version()
if _ver["sig"] != _sig:
    for _m in [m for m in sys.modules
               if m == "spy_option_screener" or m.startswith("spy_option_screener.")]:
        del sys.modules[_m]
    _ver["sig"] = _sig

from spy_option_screener import _env
_env.load()   # pick up SLACK_WEBHOOK_URL / POLYGON_API_KEY from .env

from spy_option_screener.data import loader
from spy_option_screener.data import market_calendar as mcal
from spy_option_screener.data import tracker_store as ts
from spy_option_screener.screener import gap_screener as gap_mod

st.set_page_config(page_title="SPY Gap Screener", layout="wide", page_icon="📊")

ET = ZoneInfo("America/New_York")
HIST_START = "2000-01-01"
REFRESH_EVERY = "60s"   # the live parts re-check on their own at this pace


@st.cache_data(ttl=180, show_spinner=False)
def _download_fresh(expected: str) -> None:
    """Re-download the daily history. Cached for 3 minutes per target date so
    a late-arriving bar doesn't trigger a full download every refresh."""
    loader.load_history(start=HIST_START, refresh=True)


def history_up_to_date() -> pd.DataFrame:
    """Daily history, re-downloaded whenever it's behind where it should be:
    today's FINAL bar once the session is settled, otherwise the prior
    session's. (Before this, today's finished bar never loaded until the next
    day, so the page kept showing the previous close -- and an old alert.)"""
    now_et = dt.datetime.now(ET)
    today = now_et.date()
    expected = (today if (mcal.is_trading_day(today)
                          and now_et.time() >= loader.SESSION_FINAL_TIME)
                else mcal.prev_trading_day(today))
    hist = loader.load_history(start=HIST_START)
    if pd.Timestamp(hist.index[-1]).date() < expected:
        _download_fresh(str(expected))
        hist = loader.load_history(start=HIST_START)
    return hist


@st.cache_resource(ttl=45, show_spinner=False)
def get_view():
    return gap_mod.build_view(hist_start=HIST_START)


@st.cache_data(ttl=120, show_spinner=False)
def get_tracked():
    """Sheet rows with signal status refreshed against real prices, including
    today's in-progress bar, so a target/stop touched this session shows now."""
    client = ts.client_from_info(st.secrets["gcp_service_account"])
    ws = ts.get_worksheet(client, st.secrets["gsheet"]["url"])
    daily = loader.with_live_bar(loader.load_history(start=HIST_START))
    ts.refresh_statuses(ws, daily)
    return ts.list_entries(ws)


def spy_header(now_et: dt.datetime) -> tuple[float, str]:
    today = now_et.date()
    is_trading_today = mcal.is_trading_day(today)
    open_now = is_trading_today and dt.time(9, 30) <= now_et.time() <= dt.time(16, 0)

    hist = history_up_to_date()
    last_bar = pd.Timestamp(hist.index[-1]).date()
    last_close = float(hist["close"].iloc[-1])

    if open_now:
        live_px = loader.live_spot()
        return (live_px if live_px else last_close), "🟢 Live"
    nxt = today if (is_trading_today and now_et.time() < dt.time(9, 30)) \
        else mcal.next_trading_day(today)
    return last_close, (f"🔒 Markets closed — last close {last_bar:%a %b %d}, "
                        f"next session {nxt:%a %b %d}")


PREVIEW_PROFIT_PCT, PREVIEW_LOSS_PCT = 0.45, 0.50
HOLD_DAYS = {"~1 week": 7, "~2 weeks": 14, "~1 month": 30}


def render_contract_picker(view) -> None:
    """LAYOUT PREVIEW. The controls respond, but every dollar figure is a
    placeholder (a flat +45% / -50% of the budget midpoint). Nothing sits
    behind this yet -- no option-chain feed -- so it is not a real quote or a
    real recommendation, and the page says so."""
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(":material/tune: Pick a contract")
            st.badge("Preview · not live yet", icon=":material/construction:",
                     color="orange")
        st.caption("Tell us what you'd realistically spend on one contract and "
                   "we'll suggest one that fits this alert, with what you could "
                   "make or lose.")

        c1, c2 = st.columns([3, 2])
        lo, hi = c1.slider("What would you spend on one contract?",
                           min_value=100, max_value=1500, value=(300, 700),
                           step=50, format="$%d", key="pick_budget")
        hold = c2.segmented_control("How long to hold", list(HOLD_DAYS),
                                    default="~2 weeks", key="pick_hold") or "~2 weeks"

        days = HOLD_DAYS[hold]
        expiry, _ = mcal.next_weekly_expiry(dt.date.today(), min_dte=days,
                                            max_dte=days + 5)
        cost = int(round((lo + hi) / 2 / 5) * 5)
        profit, loss = round(cost * PREVIEW_PROFIT_PCT), round(cost * PREVIEW_LOSS_PCT)

        if view.qualified:
            side = "PUT" if view.gap_type == "up" else "CALL"
            where = f"\\${round(view.target):d}"
            levels = (f"SPY target \\${view.target:.2f} · stop \\${view.stop:.2f}")
        else:
            side, where = "CALL or PUT", "near the target"
            levels = "No alert is live, so this is just an example of the layout."

        st.space("small")
        st.markdown(f"**Suggested contract** &nbsp; SPY {where} {side} · "
                    f"exp {expiry:%b %d}")
        st.caption(levels)
        m1, m2, m3 = st.columns(3)
        m1.metric("You pay", f"${cost:,}", help="Per contract (100 shares)")
        m2.metric("If SPY reaches the target", f"+${profit:,}",
                  f"+{PREVIEW_PROFIT_PCT:.0%}")
        m3.metric("If SPY hits the stop", f"-${loss:,}",
                  f"-{PREVIEW_LOSS_PCT:.0%}")
        st.caption("Placeholder numbers, shown only to preview the layout. "
                   "They are not calculated from real option prices yet. "
                   "Options can lose their full cost; the stop is where you'd "
                   "choose to exit, not a guarantee.")


def render_signal() -> None:
    try:
        view = get_view()
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't check for a gap: {e}")
        return

    if not view.qualified:
        st.markdown("### 😴 No open gap right now")
        st.caption(view.setup)
        st.caption("Most days have no qualifying gap sitting open — that's "
                   "expected. You'll see levels here the next time one is live.")
    else:
        arrow = "📉➡️📈" if view.gap_type == "up" else "📈➡️📉"
        st.markdown(f"### {arrow}  Gap {view.gap_type.upper()}  ·  {view.gap_date}")
        est = "  _(estimated)_" if view.spot_estimated else ""
        st.caption(f"day {view.days_elapsed} of 14{est}")

        c1, c2 = st.columns(2)
        c1.metric("TARGET", f"${view.target:.2f}", f"{view.target_pct:+.1%}",
                  help="The pre-gap price level — where this gap closes")
        c2.metric("STOP", f"${view.stop:.2f}", f"{view.stop_pct:+.1%}",
                  delta_color="inverse",
                  help="2.0x ATR beyond the gap open — cuts the read if "
                       "price runs this far the wrong way")
        st.metric("TIME STOP", view.time_stop_date,
                  f"{view.days_left} sessions left", delta_color="off")

        st.caption(f"historical fill rate for this type of gap: "
                   f"**{view.historical_fill_rate:.0%}** within 14 trading days")

    render_contract_picker(view)

    st.divider()
    st.caption(view.caveat)
    st.caption("CLI: `python gap_alert.py`  ·  scheduled alerts: "
               "`.github/workflows/spy-alert.yml`")


def render_tracker() -> None:
    st.caption("Every contract tracked against a gap signal. TARGET / STOP / "
               "TIME STOP are checked automatically against real SPY price "
               "data — that's the screener's own accuracy. Entry/exit price "
               "and P&L come from real brokerage screenshots, not an "
               "estimate — that's the real trade result.")
    try:
        df = get_tracked()
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't reach the tracking sheet: {e}")
        return

    if df.empty:
        st.info("Nothing tracked yet.")
        return

    hit, n_res = ts.signal_accuracy(df)
    if n_res:
        st.metric("Screener accuracy (resolved signals)",
                  f"{hit}/{n_res} hit target ({hit/n_res:.0%})",
                  help="Counted per distinct gap signal — several contracts "
                       "tracking the same gap count once.")
        if n_res < 20:
            st.caption(f"Only {n_res} resolved so far — the backtest says ~85% "
                       "(26 years, 4,776 gaps); too few here to compare yet.")

    # Two separate tracks: Status = how YOUR contract did (Win/Loss from real
    # P&L once sold); Signal = how the SCREENER's call did (target / stop /
    # time-out against real SPY prices). Only the signal feeds the accuracy
    # score above.
    result_badge = {"win": "🏆 Win", "loss": "❌ Loss", "flat": "➖ Flat"}
    position_badge = {"watching": "👀 watching", "open": "🟡 open"}
    signal_badge = {"open": "⏳ pending", "watching": "⏳ pending",
                    "hit_target": "🟢 hit target", "hit_stop": "🔴 hit stop",
                    "timed_out": "⚪ timed out"}
    show = df.copy()
    show["status_view"] = [
        result_badge.get(ts.trade_result(e, p))
        or position_badge.get(s, "— not sold")
        for e, p, s in zip(df["exit_price"], df["pnl_dollars"], df["status"])]
    show["signal_view"] = df["status"].map(lambda s: signal_badge.get(s, s))
    for c in ["entry_price", "target", "stop", "exit_price"]:
        show[c] = pd.to_numeric(show[c], errors="coerce").map(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    show["pnl_dollars"] = pd.to_numeric(df["pnl_dollars"], errors="coerce").map(
        lambda x: f"${x:+,.0f}" if pd.notna(x) else "")
    show["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce").map(
        lambda x: f"{x:+.1%}" if pd.notna(x) else "")

    st.dataframe(
        show[["date_added", "contract", "status_view", "pnl_dollars",
              "pnl_pct", "entry_price", "exit_price", "exit_date",
              "gap_type", "target", "stop", "time_stop_date",
              "signal_view", "resolution_date", "notes"]],
        hide_index=True, width="stretch",
        column_config={
            "date_added": "Added", "contract": "Contract",
            "status_view": "Status",
            "pnl_dollars": "P&L $", "pnl_pct": "P&L %",
            "entry_price": "Entry", "exit_price": "Sold at",
            "exit_date": "Sold on", "gap_type": "Gap",
            "target": "Target", "stop": "Stop",
            "time_stop_date": "Time stop",
            "signal_view": "Signal", "resolution_date": "Signal resolved",
            "notes": "Notes",
        })


@st.fragment(run_every=REFRESH_EVERY)
def live_page() -> None:
    """Everything that can change while the page sits open. Reruns itself on
    a timer, so an old alert can't stay on screen."""
    now_et = dt.datetime.now(ET)
    spot, status = spy_header(now_et)
    st.markdown(f"## SPY&nbsp;&nbsp;\\${spot:,.2f}")
    st.caption(status)
    st.caption("🔒 Manual only — this page never places a trade. You check it, "
               "you decide.")
    st.caption(f"🔄 Updated {now_et:%-I:%M:%S %p} ET — refreshes itself every minute")
    st.divider()

    tab_signal, tab_tracker = st.tabs(["📊 Signal", "📋 Tracker"])
    with tab_signal:
        render_signal()
    with tab_tracker:
        render_tracker()


live_page()
