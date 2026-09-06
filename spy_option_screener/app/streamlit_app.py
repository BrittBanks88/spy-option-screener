"""SPY Weekly Option Screener -- Robinhood-style options chain.

    streamlit run spy_option_screener/app/streamlit_app.py

Option Chain tab : pick Call/Put + an expiration, scroll the strike list the
                   way a broker shows it, tap a strike for breakeven, chance of
                   profit and Greeks. The extra column brokers don't give you
                   is Score -- the screener's 0-100 rank of how good that
                   contract is to BUY.
Backtest tab     : test a rule-based version on reconstructed history.

Educational tool. Reconstructed prices are approximate. Not investment advice.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from spy_option_screener import _env
_env.load()   # pick up SLACK_WEBHOOK_URL / POLYGON_API_KEY from .env

from spy_option_screener.data import loader
from spy_option_screener.signals import strategies as st_mod
from spy_option_screener.backtest import engine, metrics
from spy_option_screener.backtest import intraday as bt_intraday
from spy_option_screener.backtest.rules import TradeRules
from spy_option_screener.screener import chain as chain_mod
from spy_option_screener.screener import score as score_mod
from spy_option_screener.screener.daily_pick import todays_pick
from spy_option_screener.screener import trade_plan as tp_mod
from spy_option_screener.data import market_calendar as mcal

st.set_page_config(page_title="SPY Weekly Option Screener", layout="wide",
                   page_icon="📈")


# ── data helpers ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_history(start, refresh=False):
    return loader.load_history(start=start, refresh=refresh)


@st.cache_data(ttl=600)
def get_live_chain(dte_max):
    ch = loader.load_live_chain(dte_max=dte_max)
    return ch, float(ch.attrs["spot"])


def upcoming_expiries(n=6):
    """Next ``n`` SPY expirations (every trading day; holidays skipped)."""
    out, d = [], dt.date.today()
    while len(out) < n:
        d = mcal.next_trading_day(d)
        out.append(d)
    return out


@st.cache_data(ttl=900)
def get_daily_pick(interval, entry_time, buy_zone):
    return todays_pick(interval=interval, entry_time=entry_time, buy_zone=buy_zone)


@st.cache_data(ttl=900)
def get_trade_plan(entry_time, interval, for_date):
    return tp_mod.build_plan(entry_time=entry_time, interval=interval,
                             for_date=for_date or None)


def fmt_pct(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{d}%}"


def fmt_money(x):
    return "–" if x is None or not np.isfinite(x) else f"${x:,.0f}"


from spy_option_screener.data import polygon as _poly

# ── sidebar (advanced knobs, hidden from the main flow) ─────────────────────
with st.sidebar:
    st.header("Settings")
    _srcs = ["Reconstructed from VIX", "Live (yfinance, delayed)"]
    if _poly.available():
        _srcs.insert(0, "Polygon (real-time)")
    chain_src = st.radio(
        "Chain data", _srcs, index=0,
        help="Polygon = real option chain with Greeks (needs POLYGON_API_KEY). "
             "Reconstructed always works and matches the backtest engine. "
             "yfinance quotes are unreliable outside market hours / near 0-DTE.")
    if not _poly.available():
        st.caption("💡 set `POLYGON_API_KEY` in `.env` for real-time chains")
    hist_start = st.selectbox("History window", ["2018-01-01", "2015-01-01",
                              "2012-01-01", "2020-01-01"], index=0)
    rv_note = st.empty()
    st.divider()
    st.caption("Score weights")
    w_be = st.slider("Breakeven vs expected move", 0.0, 1.0, 0.30, 0.05)
    w_dl = st.slider("Delta (participation)", 0.0, 1.0, 0.15, 0.05)
    w_th = st.slider("Theta bleed", 0.0, 1.0, 0.20, 0.05)
    w_vv = st.slider("Vol value (IV vs realized)", 0.0, 1.0, 0.20, 0.05)
    w_lq = st.slider("Liquidity", 0.0, 1.0, 0.15, 0.05)
    weights = {"breakeven": w_be, "delta": w_dl, "theta": w_th,
               "vol_value": w_vv, "liquidity": w_lq}

hist = get_history(hist_start)
spot_hist = float(hist["close"].iloc[-1])
rv = float(hist["realized_vol_20"].iloc[-1])
vix_now = float(hist["vix"].iloc[-1])
vix_base = float(hist["vix"].tail(40).mean())
day_chg = float(hist["ret"].iloc[-1])
last_bar = pd.Timestamp(hist.index[-1]).date()
_today = dt.date.today()
_open = _today == last_bar and mcal.is_trading_day(_today)
if mcal.is_trading_day(_today) and _today > last_bar:
    _next_sess = _today
else:
    _next_sess = mcal.next_trading_day(max(_today, last_bar))
last_label = "Today" if _open else f"Last close ({last_bar:%a %b %d})"
rv_note.caption(f"SPY realized vol (20d): **{rv:.1%}**  ·  VIX: **{vix_now:.1f}**")
if not _open:
    st.caption(f"🔒 Markets closed — showing the {last_bar:%A %b %d} close. "
               f"Next session: {_next_sess:%A %b %d}.")

# current momentum read, shown as a hint
sig_hist = st_mod.combine(hist).iloc[-1]
sig_txt = {1: "🟢 leaning bullish", -1: "🔴 leaning bearish",
           0: "⚪ no clear lean"}[int(sig_hist["direction"])]

# ── header ─────────────────────────────────────────────────────────────────
h1, h2, h3 = st.columns([2, 1, 1])
h1.markdown(f"## SPY&nbsp;&nbsp;\\${spot_hist:,.2f}")
h2.metric(last_label, fmt_pct(day_chg, 2))
h3.metric("Signal", sig_txt.split(" ", 1)[1], help="Blend of pullback, "
          "trend-continuation, RSI-2 mean reversion, breakout and vol-regime "
          "signals on daily SPY. A hint, not a recommendation — you choose "
          "Call or Put below.")

tab_pick, tab_plan, tab_chain, tab_bt, tab_about = st.tabs(
    ["⭐ Today's Pick", "📋 Trade Plan (Slack)", "🔗 Option Chain", "📊 Backtest",
     "❓ How to use this"])

# ═══════════════════════════════════════════════════════════════════════════
# TRADE PLAN  —  pullback → continuation, formatted for Slack
# ═══════════════════════════════════════════════════════════════════════════
with tab_plan:
    st.caption("Looks for a **pullback inside an established trend** (the signal "
               "with the best backtest), then writes a full plan — entry / stop / "
               "target / runner / time stop — ready to paste into Slack.")
    pl1, pl2, pl3 = st.columns(3)
    tp_entry = pl1.selectbox("Entry time (ET)", ["10:00", "10:30", "11:00"], key="tp_e")
    tp_interval = pl2.selectbox("Intraday bars", ["1h", "30m"], key="tp_i")
    tp_for = pl3.text_input("Back-check a past session (YYYY-MM-DD)", "",
                            help="Leave blank for the latest session.")

    if st.button("Build trade plan", type="primary"):
        try:
            st.session_state["plan"] = get_trade_plan(tp_entry, tp_interval, tp_for.strip())
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not build the plan: {e}")

    if "plan" in st.session_state:
        pl = st.session_state["plan"]
        if not pl.qualified:
            st.warning(f"**No pullback-before-continuation setup.**  {pl.setup}")
            st.caption(pl.day_note)
        else:
            side = "CALL" if pl.direction > 0 else "PUT"
            st.success(f"### SPY ${pl.strike:.0f} {side} · exp {pl.expiry} "
                       f"({pl.dte} DTE)".replace("$", "\\$"))
            g = st.columns(4)
            g[0].metric("Entry (limit)", f"${pl.o_entry:.2f}")
            g[1].metric("Stop", f"${pl.o_stop:.2f}", "-50%")
            g[2].metric("Target 1", f"${pl.o_t1:.2f}", f"{pl.o_t1_pct:+.0%}")
            g[3].metric("Runner", f"${pl.o_run:.2f}", f"{pl.o_run_pct:+.0%}")
            g = st.columns(4)
            g[0].metric("SPY entry", f"${pl.u_entry:.2f}"
                        + (" (est)" if pl.u_entry_estimated else ""))
            g[1].metric("SPY stop", f"${pl.u_stop:.2f}")
            g[2].metric("SPY T1 / runner", f"${pl.u_t1:.0f} / ${pl.u_run:.0f}")
            g[3].metric("Reward:risk (T1)", f"{pl.reward_risk:.1f} : 1")
            if pl.entry_quality:
                st.warning(f"⚠️ {pl.entry_quality}")
            st.caption(f"trigger: SPY green & { 'above' if pl.direction>0 else 'below'} "
                       f"${pl.entry_hold_level:.2f} — {pl.entry_window}  ·  "
                       f"time stop: {pl.time_stop}")
            if pl.overnight:
                st.caption("🌙 " + pl.overnight.replace("*", ""))

        st.markdown("**Paste into Slack** (simple — copy button top-right):")
        st.code(pl.slack_simple_text(), language=None)
        with st.expander("Detailed version"):
            st.code(pl.slack_text(), language=None)
        with st.expander("Slack Block Kit payload (for an incoming webhook / bot)"):
            import json as _json
            st.code(_json.dumps(pl.slack_payload(style="simple"), indent=2,
                                ensure_ascii=False), language="json")
        st.caption("CLI: `python slack_plan.py` · `--json` for the payload · "
                   "`--webhook <URL>` to POST it (sends a message — opt-in).  "
                   "Scheduled: `python run_alert.py --mode auto` (see "
                   "`.github/workflows/spy-alert.yml`).")
        st.info("Backtest of this exact plan (60 setups, 2024–2026, bull market): "
                "win rate ~53%, avg trade ≈ +7% of premium, profit factor ≈ 1.2, "
                "expectancy ≈ +0.13 R. The first positive result in the project — "
                "but a short, one-regime sample. `python research/backtest_trade_plan.py`.")
        st.caption("🌙 On **overnight holds:** buying an option just to hold one "
                   "night loses in backtest (≈ −4% to −10%, "
                   "`research/backtest_overnight.py`) — the spread + a night of "
                   "theta swamp the ~3–5 bps of index drift. The 🌙 line above "
                   "only rates carrying *this* position, which is already an edge.")

# ═══════════════════════════════════════════════════════════════════════════
# TODAY'S PICK  (gap + opening-range-breakout + daily bias -> one contract)
# ═══════════════════════════════════════════════════════════════════════════
with tab_pick:
    st.caption("Combines the opening **gap-continuation** and **opening-range-"
               "breakout** intraday signals with the daily trend/mean-reversion "
               "bias, then ranks the nearest weekly chain. Price is not a filter.")
    pc1, pc2, pc3 = st.columns(3)
    pk_interval = pc1.selectbox("Intraday bars", ["30m", "15m", "1h"], index=0)
    pk_entry = pc2.selectbox("Decide / enter at (ET)", ["10:00", "10:30", "11:00"],
                             index=1)
    pk_zone = pc3.checkbox("Directional-buy zone only", value=True)

    if st.button("Get today's pick", type="primary"):
        try:
            st.session_state["pick"] = get_daily_pick(pk_interval, pk_entry, pk_zone)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not build the pick: {e}")

    if "pick" in st.session_state:
        p = st.session_state["pick"]
        if p.live:
            st.success(f"🟢 {p.day_note}")
        else:
            st.warning(f"🔒 {p.day_note}")
        st.caption(f"signal source: **{p.session_date}** session · as of {p.asof}")
        k = st.columns(4)
        k[0].metric("SPY (at entry time)", f"${p.spot:,.2f}")
        k[1].metric("Opening gap", f"{p.gap:+.2%}")
        k[2].metric("VIX", f"{p.vix:.1f}")
        k[3].metric("Conviction", f"{p.conviction:.0%}")

        headline = p.headline().replace("$", "\\$")
        if p.direction == 0 or p.contract is None:
            st.warning(headline)
        else:
            st.success("### " + headline)

        st.markdown("**Why:**")
        for r in p.reasons:
            st.markdown(f"- {r}")

        cA, cB = st.columns(2)
        for col, label, c, want in [(cA, "Best CALL", p.call_alt, 1),
                                    (cB, "Best PUT", p.put_alt, -1)]:
            with col:
                chosen = " ← pick" if p.direction == want and p.contract else ""
                st.markdown(f"#### {label}{chosen}")
                if not c:
                    st.write("— none in the buy zone —")
                    continue
                st.markdown(
                    f"**\\${c['strike']:.0f} {c['type'].upper()}** · exp {p.expiry} "
                    f"({p.dte}d)  \n"
                    f"Premium ~\\${c['mid']:.2f}  (\\${c['mid']*100:,.0f}/contract)  \n"
                    f"Breakeven \\${c['breakeven']:.2f} ({c['be_pct']:+.2%})  \n"
                    f"Chance of profit {c['chance_of_profit']:.0%} · "
                    f"leverage {c['leverage']:.0f}×  \n"
                    f"Δ {c['delta']:+.2f} · θ -\\${abs(c['theta'])*100:.0f}/day · "
                    f"IV {c['iv']:.1%}  \n"
                    f"**Score {c['score']:.0f}/100**")

        st.info(p.note)
        st.caption("Backtest of these signals (2-yr sample): gap-continuation "
                   "cut the long-option bleed roughly in half vs the daily "
                   "signals; opening-range-breakout is not yet proven on free "
                   "data. Still net-negative — see the Backtest tab.")

# ═══════════════════════════════════════════════════════════════════════════
# OPTION CHAIN  (the screener, Robinhood-style)
# ═══════════════════════════════════════════════════════════════════════════
with tab_chain:
    top = st.container()

    c1, c2 = st.columns(2)
    opt_type = c1.segmented_control("Type", ["Call", "Put"], default="Call",
                                    selection_mode="single")
    action = c2.segmented_control("Action", ["Buy", "Sell"], default="Buy",
                                  selection_mode="single")
    opt_type = opt_type or "Call"
    action = action or "Buy"
    direction = 1 if opt_type == "Call" else -1

    exps = upcoming_expiries(6)
    labels = [f"{d:%a} {d.month}/{d.day}  ·  {(d - dt.date.today()).days}d"
              for d in exps]
    pick_label = st.radio("Expiration", labels, horizontal=True,
                          index=min(2, len(labels) - 1))
    exp_date = exps[labels.index(pick_label)]
    dte = max((exp_date - dt.date.today()).days, 1)

    sc1, sc2 = st.columns([3, 2])
    strike_band = sc1.slider("Strike range (± % of spot)", 1, 8, 3) / 100.0
    focus = sc2.checkbox("Directional-buy zone only", value=True,
                         help="Hide deep in-the-money strikes (expensive, "
                              "little leverage) and far out-of-the-money "
                              "lottery tickets. Keeps |delta| ≈ 0.20–0.62.")

    # ── build + score the chain ──────────────────────────────────────────
    try:
        if chain_src.startswith("Polygon"):
            raw = _poly.option_chain(exp_date, strike_band=strike_band + 0.01)
            chain = chain_mod.prep_polygon_chain(raw, live_spot=loader.live_spot())
            spot = float(chain.attrs["spot"])
            age = chain.attrs.get("quote_age_seconds")
            if chain.attrs.get("reanchored"):
                st.caption(f"📡 Polygon chain quote was {age:.0f}s old — Greeks "
                           f"re-priced to the live SPY quote "
                           f"(${chain.attrs['snapshot_spot']:.2f} → ${spot:.2f})")
            elif age is not None:
                st.caption(f"📡 Polygon quote age: {age:.0f}s (real-time)")
        elif chain_src.startswith("Live"):
            raw, spot = get_live_chain(max(dte + 1, 2))
            chain = chain_mod.enrich_live_chain(raw, fallback_vix=vix_now)
            want = min(chain["dte"].unique(), key=lambda x: abs(x - dte))
            keep = dict(chain.attrs)
            chain = chain[chain["dte"] == want].copy()
            chain.attrs.update(keep)
            spot = float(chain.attrs["spot"])
        else:
            spot = spot_hist
            chain = chain_mod.synthetic_chain(spot, vix_now, dte,
                                              vix_baseline=vix_base,
                                              width=strike_band + 0.005)
        scored = score_mod.score_chain(chain, direction, spot, rv,
                                       weights=weights)
        scored = scored[(scored["strike"] >= spot * (1 - strike_band)) &
                        (scored["strike"] <= spot * (1 + strike_band))].copy()
        if focus:
            ad = scored["delta"].abs()
            scored = scored[ad.between(0.20, 0.62)].copy()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not build the chain: {e}")
        st.stop()

    if scored.empty:
        st.warning("No contracts in that strike range — widen it.")
        st.stop()

    # ── the pick ────────────────────────────────────────────────────────
    best = scored.sort_values("score", ascending=False).iloc[0]
    with top:
        verb = "buy" if action == "Buy" else "sell"
        st.info(
            f"**Screener's pick to {verb}:  SPY \\${best['strike']:.0f} "
            f"{opt_type}**  ·  score **{best['score']:.0f}/100**  ·  "
            f"premium ~\\${best['mid']:.2f} (\\${best['mid']*100:.0f} to open)  ·  "
            f"breakeven \\${best['breakeven']:.2f}  ·  "
            f"chance of profit ~{best['chance_of_profit']:.0%}"
            + ("" if action == "Buy" else
               "  —  ⚠️ score rates BUYING; a low score is the better sell."))

    # ── the chain list ─────────────────────────────────────────────────
    disp = scored.sort_values("strike", ascending=(opt_type == "Put")).copy()
    itm = (disp["strike"] < spot) if opt_type == "Call" else (disp["strike"] > spot)
    view = pd.DataFrame({
        "": np.where(itm, "● ITM", ""),
        "Strike": disp["strike"].map(lambda x: f"{x:.0f}"),
        "Premium": disp["mid"].map(lambda x: f"${x:.2f}"),
        "To open": (disp["mid"] * 100).map(lambda x: f"${x:,.0f}"),
        "Breakeven": disp["breakeven"].map(lambda x: f"${x:.2f}"),
        "% to B/E": disp["be_pct"].map(lambda x: f"{x:+.2%}"),
        "Chance of profit": disp["chance_of_profit"].map(lambda x: f"{x:.0%}"),
        "Delta": disp["delta"].map(lambda x: f"{x:+.2f}"),
        "Theta/day": (disp["theta"] * 100).map(lambda x: f"-${abs(x):.0f}"),
        "IV": disp["iv"].map(lambda x: f"{x:.1%}"),
        "Score": disp["score"],
    })
    st.caption(f"{len(view)} strikes · spot \\${spot:,.2f} · "
               f"1σ move by {exp_date:%b %d}: ±\\${best['exp_move']:.2f} "
               f"(±{best['exp_move']/spot:.1%})   —   tap a row for detail")

    event = st.dataframe(
        view.style.background_gradient(subset=["Score"], cmap="RdYlGn",
                                       vmin=0, vmax=100)
                  .format({"Score": "{:.0f}"}),
        width="stretch", hide_index=True, height=430,
        on_select="rerun", selection_mode="single-row", key="chainsel")

    rows = event.get("selection", {}).get("rows", []) if event else []
    sel = disp.iloc[rows[0]] if rows else best

    # ── detail card ────────────────────────────────────────────────────
    st.divider()
    long_ = action == "Buy"
    prem = float(sel["mid"])
    cost = prem * 100
    st.markdown(f"### {action} SPY&nbsp;\\${sel['strike']:.0f} {opt_type}"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;exp {exp_date:%a %b %d} ({dte}d)")

    m = st.columns(4)
    m[0].metric("Cost to open" if long_ else "Credit received", fmt_money(cost))
    if long_:
        m[1].metric("Max loss", fmt_money(cost))
        m[2].metric("Max profit", "Unlimited" if opt_type == "Call"
                    else fmt_money(sel["strike"] * 100 - cost))
    else:
        m[1].metric("Max profit", fmt_money(cost))
        m[2].metric("Max loss", "Very large" if opt_type == "Call"
                    else fmt_money(sel["strike"] * 100 - cost))
    m[3].metric("Breakeven", f"${sel['breakeven']:.2f}")

    m = st.columns(4)
    cop = sel["chance_of_profit"] if long_ else 1 - sel["chance_of_profit"]
    m[0].metric("Chance of profit", f"{cop:.0%}")
    m[1].metric("Move to breakeven", f"{sel['be_pct']:+.2%}")
    m[2].metric("Leverage", f"{sel['leverage']:.0f}×",
                help="Dollar exposure controlled per dollar of premium "
                     "(|delta|·spot / premium)")
    m[3].metric("Score (to buy)", f"{sel['score']:.0f}/100")

    g = st.columns(5)
    g[0].metric("Delta", f"{sel['delta']:+.3f}")
    g[1].metric("Gamma", f"{sel['gamma']:.4f}")
    g[2].metric("Theta / day", f"-${abs(sel['theta'])*100:.0f}")
    g[3].metric("Vega", f"{sel['vega']:.3f}")
    g[4].metric("Implied vol", f"{sel['iv']:.1%}",
                delta=f"{sel['iv']-rv:+.1%} vs realized",
                delta_color="inverse")

    dcard, pcard = st.columns([1, 1])
    with dcard:
        st.caption("Why this score")
        comps = {"Breakeven vs move": sel["s_breakeven"],
                 "Delta": sel["s_delta"], "Theta bleed": sel["s_theta"],
                 "Vol value (IV vs realized)": sel["s_vol_value"],
                 "Liquidity": sel["s_liquidity"]}
        for name, val in comps.items():
            st.progress(float(np.clip(val, 0, 1)), text=f"{name} — {val:.2f}")
        if sel["iv"] > rv:
            st.caption(f"⚠️ IV {sel['iv']:.1%} is above realized {rv:.1%} — "
                       "you're paying a volatility premium, the usual drag on "
                       "long options.")
    with pcard:
        S = np.linspace(spot * (1 - strike_band - 0.01),
                        spot * (1 + strike_band + 0.01), 240)
        intrinsic = (np.maximum(S - sel["strike"], 0) if opt_type == "Call"
                     else np.maximum(sel["strike"] - S, 0))
        pl = (intrinsic - prem) * 100 * (1 if long_ else -1)
        fig = go.Figure(go.Scatter(x=S, y=pl, fill="tozeroy",
                                   line=dict(color="#2E7D32")))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.add_vline(x=spot, line_dash="dot", annotation_text="spot")
        fig.add_vline(x=sel["breakeven"], line_dash="dot",
                      line_color="#C62828", annotation_text="B/E")
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                          title="Profit / loss at expiration",
                          xaxis_title="SPY price", yaxis_title="$ per contract")
        st.plotly_chart(fig, width="stretch")

    st.download_button("Download full ranked chain (CSV)",
                       scored.to_csv(index=False),
                       f"spy_{opt_type.lower()}s_{exp_date}.csv")

# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.caption("Test a mechanical version: on each signal, buy the nearest "
               "weekly at a target delta, manage with a profit target / stop.")

    mode = st.segmented_control(
        "Signal engine", ["Daily indicators", "Intraday (gap + ORB, timed entry)"],
        default="Daily indicators", selection_mode="single")
    intraday_mode = (mode or "Daily indicators").startswith("Intraday")

    b = st.columns(4)
    if intraday_mode:
        strat = b[0].selectbox("Intraday bars", ["1h (≈2 yr)", "30m (≈60 d)"],
                               index=0)
        bt_interval = "1h" if strat.startswith("1h") else "30m"
        bt_entry = b[1].selectbox("Entry time (ET)", ["10:00", "10:30", "11:00"])
        use_bias = b[2].checkbox("Blend daily bias", value=True)
        capital = b[3].number_input("Capital ($)", 5_000, 500_000, 25_000, 5_000)
    else:
        strat = b[0].selectbox("Signal", list(st_mod.STRATEGIES) + ["combo"], index=3)
        capital = b[1].number_input("Capital ($)", 5_000, 500_000, 25_000, 5_000)
        bt_entry = b[2].selectbox("Entry time (ET)", ["close", "10:00", "10:30", "11:00"])
        b[3].caption(" ")

    b = st.columns(4)
    tgt_delta = b[0].slider("Target delta", 0.15, 0.75, 0.45, 0.05)
    risk = b[1].slider("Premium risk / trade", 0.01, 0.10, 0.03, 0.01)
    tgt_dte = b[2].slider("Target DTE", 1, 9, 5)
    pt = b[3].slider("Profit target (× premium)", 0.25, 3.0, 1.0, 0.25)
    sl = st.slider("Stop loss (× premium)", 0.2, 0.9, 0.6, 0.1)

    if st.button("Run backtest", type="primary"):
        with st.spinner("Simulating…"):
            rules = TradeRules(target_dte=tgt_dte, target_delta=tgt_delta,
                               risk_per_trade=risk, profit_target_pct=pt,
                               stop_loss_pct=sl,
                               entry_time=None if bt_entry == "close" else bt_entry)
            if intraday_mode:
                equity, trades, _, _, _ = bt_intraday.run_intraday(
                    interval=bt_interval, entry_time=bt_entry,
                    use_daily_bias=use_bias, rules=rules,
                    starting_capital=capital, hist_start=hist_start)
                df = get_history(hist_start)
            else:
                df = get_history(hist_start)
                sig = (st_mod.combine(df) if strat == "combo"
                       else st_mod.STRATEGIES[strat](df))
                if bt_entry == "close":
                    equity, trades, _ = engine.run(df, sig, rules,
                                                   starting_capital=capital)
                else:
                    from spy_option_screener.data import intraday as _it
                    _bars = _it.load_intraday("1h")
                    _feat = _it.daily_features(_bars, df, or_minutes=60,
                                               signal_cutoff=bt_entry,
                                               entry_times=(bt_entry, "10:00",
                                                            "10:30", "11:00"))
                    _epx = _feat[f"px_{bt_entry.replace(':', '')}"].dropna()
                    _dd = df.reindex(_feat.index).dropna(subset=["close", "vix"])
                    equity, trades, _ = engine.run(
                        _dd, sig.reindex(_dd.index), rules,
                        starting_capital=capital, warmup=40,
                        entry_prices=_epx.reindex(_dd.index))
            summ = metrics.summary(equity, trades)
        st.session_state["bt"] = (df, equity, trades, summ, capital)

    if "bt" in st.session_state:
        df, equity, trades, summ, cap0 = st.session_state["bt"]
        k = st.columns(6)
        k[0].metric("Total return", fmt_pct(summ.get("total_return")))
        k[1].metric("CAGR", fmt_pct(summ.get("cagr")))
        k[2].metric("Max drawdown", fmt_pct(summ.get("max_drawdown")))
        k[3].metric("Sharpe", f"{summ.get('sharpe', float('nan')):.2f}")
        k[4].metric("Win rate", fmt_pct(summ.get("win_rate")))
        k[5].metric("Trades", f"{summ.get('n_trades', 0)}")

        bh = df["close"] / df["close"].reindex(equity.index).iloc[0] * cap0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Strategy"))
        fig.add_trace(go.Scatter(x=equity.index,
                                 y=bh.reindex(equity.index).values,
                                 name="SPY buy & hold", line=dict(dash="dot")))
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_type="log", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

        if summ.get("total_return", 0) < 0:
            st.warning("Negative — SPY's volatility risk premium works against a "
                       "naive long-option buyer. Tune the signal / exits, or use "
                       "this to size a view you already hold rather than as a "
                       "standalone entry.")
        if not trades.empty:
            show = trades.copy()
            show["return_pct"] = show["return_pct"].map(lambda x: f"{x:.0%}")
            st.dataframe(show, width="stretch", height=280)

# ═══════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(f"""
## How to use this

It works like a broker's options screen, with one extra column.

**1. Pick a direction.** Tap **Call** if you think SPY goes up by
{exp_date:%b %d}, **Put** if down. (The *Signal* chip up top blends the
pullback / trend / mean-reversion indicators — a hint, you decide.)

**2. Pick an expiration.** The pills are the next few weekly expirations with
days-to-expiry. Closer = cheaper but less time to be right.

**3. Read the chain.** Each row is one strike:

| Column | What it tells you |
|---|---|
| **Premium / To open** | price per share, and dollars for one contract (×100) |
| **Breakeven** | SPY must pass this at expiration for the trade to make money |
| **% to B/E** | how far SPY has to move from here to break even |
| **Chance of profit** | model probability SPY finishes past breakeven |
| **Delta** | ≈ how much the option moves per \\$1 of SPY; loose "chance it finishes in the money" |
| **Theta/day** | dollars of time-decay you lose each day you hold |
| **IV** | implied volatility priced into this contract |
| **Score** | 0–100 — the screener's rank of how good this is **to buy** (green = better) |

**4. Tap a row** for the full card: max loss, leverage, all the Greeks, a
profit/loss chart, and a breakdown of *why* it scored what it did.

### What the Score rewards

| Part | Weight | Good when |
|---|---|---|
| Breakeven vs expected move | {w_be:.0%} | breakeven sits inside a normal weekly move |
| Delta | {w_dl:.0%} | ~0.45 — real participation, not a lottery ticket |
| Theta bleed | {w_th:.0%} | low daily decay relative to premium |
| Vol value | {w_vv:.0%} | IV is cheap vs SPY's recent realized volatility |
| Liquidity | {w_lq:.0%} | tight bid/ask, open interest |

Adjust the weights in the sidebar.

### The signals (Today's Pick + Backtest)

| Signal | Idea | Backtest note |
|---|---|---|
| **pullback** | buy the dip inside an uptrend, once it turns back up | best-behaved — small loss over the last ~2 yr entered ~10:00 ET, vs a big loss held from the close |
| **trend_continuation** | trend intact, price makes a fresh leg | chases; a later entry makes it worse |
| **mean_reversion** | RSI-2 snapback from an extreme | "least bad" of the pure daily signals |
| **momentum_breakout** | fresh N-day range break with the trend | buys into peak IV; weakest |
| **vol_regime** | long premium when IV is cheap vs realized | occasional convexity |
| **gap_continuation** | big opening gap up → continues | ~halves the bleed vs daily-only |
| **opening_range_breakout** | first hold beyond the 9:30–10:00 range | under-validated on free data; low weight |

**pullback + mean_reversion, entered ~10:00 ET**, is the only combination that
came out roughly flat (rather than losing) in testing — on a ~2-year
bull-market sample of ~45 trades. Treat that as "promising," not "proven."

### The honest caveat

Historically SPY options have been priced *above* the move that actually
followed ~80–85% of weeks. Almost every backtest of weekly buying here is
negative. Use the screener to find the **least-overpriced** contract for a
view you already have — not as a reason to trade by itself. Reconstructed
prices are approximate (±10–20%). Intraday history is only ~2 years.
**Not investment advice.**
""")
