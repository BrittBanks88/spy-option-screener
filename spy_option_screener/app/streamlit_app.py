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

from spy_option_screener import _env
_env.load()   # pick up SLACK_WEBHOOK_URL / POLYGON_API_KEY from .env

from spy_option_screener.data import loader
from spy_option_screener.data import market_calendar as mcal
from spy_option_screener.data import tracker_store as ts
from spy_option_screener.screener import gap_screener as gap_mod

st.set_page_config(page_title="SPY Gap Screener", layout="centered", page_icon="📊")

ET = ZoneInfo("America/New_York")
HIST_START = "2000-01-01"


@st.cache_data(ttl=1800)
def get_view():
    # refresh=True: this is a live signal page, not a backtest -- always pull
    # current daily bars rather than risk a stale cached session.
    return gap_mod.build_view(hist_start=HIST_START, refresh=True)


def spy_header():
    """SPY price + a status line, auto-refreshing the cached daily history
    once it's behind where it should be (see the staleness bug this project
    hit before: a stale cache made the app claim "markets closed" all day)."""
    now_et = dt.datetime.now(ET)
    today = now_et.date()
    is_trading_today = mcal.is_trading_day(today)
    open_now = is_trading_today and dt.time(9, 30) <= now_et.time() <= dt.time(16, 0)

    expected = (today if (is_trading_today and now_et.time() >= dt.time(16, 5))
                else mcal.prev_trading_day(today))
    hist = loader.load_history(start=HIST_START)
    if pd.Timestamp(hist.index[-1]).date() < expected:
        hist = loader.load_history(start=HIST_START, refresh=True)
    last_bar = pd.Timestamp(hist.index[-1]).date()
    last_close = float(hist["close"].iloc[-1])

    if open_now:
        live_px = loader.live_spot()
        spot = live_px if live_px else last_close
        status = "🟢 Live"
    else:
        spot = last_close
        nxt = today if (is_trading_today and now_et.time() < dt.time(9, 30)) \
            else mcal.next_trading_day(today)
        status = f"🔒 Markets closed — last close {last_bar:%a %b %d}, next session {nxt:%a %b %d}"
    return spot, status


@st.cache_data(ttl=300)
def get_tracked(_daily_last_date: str):
    """Cache key includes the last daily bar date so a new close forces a
    fresh status check; gspread calls themselves aren't cached beyond that."""
    client = ts.client_from_info(st.secrets["gcp_service_account"])
    ws = ts.get_worksheet(client, st.secrets["gsheet"]["url"])
    daily = loader.load_history(start=HIST_START)
    ts.refresh_statuses(ws, daily)
    return ts.list_entries(ws)


spot, status = spy_header()
st.markdown(f"## SPY&nbsp;&nbsp;\\${spot:,.2f}")
st.caption(status)
st.caption("🔒 Manual only — this page never places a trade. You check it, you decide.")
st.divider()

tab_signal, tab_tracker = st.tabs(["📊 Signal", "📋 Tracker"])

with tab_signal:
    try:
        view = get_view()
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't check for a gap: {e}")
        st.stop()

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

    st.divider()
    st.caption(view.caveat)
    st.caption("CLI: `python gap_alert.py`  ·  scheduled alerts: "
               "`.github/workflows/spy-alert.yml`")

with tab_tracker:
    st.caption("Every contract tracked against a gap signal. TARGET / STOP / "
               "TIME STOP are checked automatically against real SPY price "
               "data — that's the screener's own accuracy. Entry/exit price "
               "and P&L come from real brokerage screenshots, not an "
               "estimate — that's the real trade result.")
    try:
        df = get_tracked(str(loader.load_history(start=HIST_START).index[-1].date()))
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't reach the tracking sheet: {e}")
        st.stop()

    if df.empty:
        st.info("Nothing tracked yet.")
    else:
        resolved = df[~df["status"].isin(ts.PENDING_STATUSES)]
        n_res = len(resolved)
        if n_res:
            hit = (resolved["status"] == "hit_target").sum()
            st.metric("Screener accuracy (resolved signals)",
                      f"{hit}/{n_res} hit target ({hit/n_res:.0%})")

        badge = {"open": "🟡 open", "watching": "👀 watching",
                 "hit_target": "🟢 hit target",
                 "hit_stop": "🔴 hit stop", "timed_out": "⚪ timed out"}
        show = df.copy()
        show["status"] = show["status"].map(lambda s: badge.get(s, s))
        for c in ["entry_price", "target", "stop", "exit_price"]:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        show["pnl_dollars"] = pd.to_numeric(df["pnl_dollars"], errors="coerce").map(
            lambda x: f"${x:+,.0f}" if pd.notna(x) else "")
        show["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce").map(
            lambda x: f"{x:+.0%}" if pd.notna(x) else "")

        st.dataframe(
            show[["date_added", "contract", "entry_price", "gap_type",
                  "target", "stop", "time_stop_date", "status",
                  "resolution_date", "exit_price", "pnl_dollars", "pnl_pct",
                  "notes"]],
            hide_index=True, width="stretch",
            column_config={
                "date_added": "Added", "contract": "Contract",
                "entry_price": "Entry", "gap_type": "Gap",
                "target": "Target", "stop": "Stop",
                "time_stop_date": "Time stop", "status": "Status",
                "resolution_date": "Resolved", "exit_price": "Exit",
                "pnl_dollars": "P&L $", "pnl_pct": "P&L %",
                "notes": "Notes",
            })
