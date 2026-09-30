"""Regime-robustness check on the pullback -> continuation TRADE PLAN.

research/backtest_trade_plan.py tested the plan on a ~2-year sample (2024-26,
bull market only) because it needs an actual 10am intraday print, and free
intraday history only goes back ~2 years. That means the ONE strategy in this
whole project with a real edge has never been checked against a correction,
a crash, or a bear market.

This script reuses the exact same signal (pullback + MA-stack trend
confirmation), the exact same ATR-based stop/target/runner levels, and the
exact same reconstructed-chain option pricing as trade_plan.py -- but swaps
the intraday 10am entry price for the NEXT day's open (the closest daily-only
stand-in), so it can run across the full cached history: 2018 correction,
2020 crash, 2022 bear market, and the 2023-25 bull run all included.

    python research/backtest_trade_plan_long.py

Not the identical plan a live alert would post (no intraday chase-check,
no live chain) -- this isolates whether the core edge (signal + levels)
survives regimes it's never been tested against. Reconstructed prices.
Not investment advice.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spy_option_screener.data import loader, market_calendar as mcal    # noqa: E402
from spy_option_screener.pricing import black_scholes as bs             # noqa: E402
from spy_option_screener.signals import strategies as dsig              # noqa: E402
from spy_option_screener.signals import indicators as ind               # noqa: E402
from spy_option_screener.screener import chain as chain_mod             # noqa: E402
from spy_option_screener.screener import score as score_mod             # noqa: E402
from spy_option_screener.screener.trade_plan import _t_years, R, Q      # noqa: E402
from research.backtest_trade_plan import outcome                        # noqa: E402

BUY_ZONE = (0.35, 0.60)
SPLIT = pd.Timestamp("2024-01-01")


def build_daily_plan(daily: pd.DataFrame, session_date) -> types.SimpleNamespace | None:
    """Same qualification + level logic as trade_plan.build_plan, but entry
    price = next day's OPEN instead of an intraday 10am print."""
    d = daily.loc[:pd.Timestamp(session_date)]
    close = d["close"]
    pb_row = dsig.pullback(d).iloc[-1]
    pdir = int(pb_row["direction"])
    if pdir == 0:
        return None
    sma20, sma50, sma100 = (ind.sma(close, 20).iloc[-1], ind.sma(close, 50).iloc[-1],
                            ind.sma(close, 100).iloc[-1])
    up_stack, dn_stack = sma20 > sma50 > sma100, sma20 < sma50 < sma100
    if not ((pdir > 0 and up_stack) or (pdir < 0 and dn_stack)):
        return None

    entry_date = mcal.next_trading_day(session_date)
    entry_ts = pd.Timestamp(entry_date)
    if entry_ts not in daily.index:
        return None
    u_entry = float(daily.loc[entry_ts, "open"])

    vix = float(d["vix"].iloc[-1])
    rv = float(d["realized_vol_20"].iloc[-1])
    atr = float(ind.atr(d["high"], d["low"], close, 14).iloc[-1])

    exp_date, dte = mcal.next_weekly_expiry(entry_date, min_dte=3, max_dte=9)
    chain = chain_mod.synthetic_chain(u_entry, vix, dte,
                                      vix_baseline=float(d["vix"].tail(40).mean()),
                                      r=R, q=Q)
    scored = score_mod.score_chain(chain, pdir, u_entry, rv, confidence=1.0)
    lo, hi = BUY_ZONE
    band = scored[scored["delta"].abs().between(lo, hi)]
    if band.empty and scored.empty:
        return None
    c = (band if not band.empty else scored).iloc[0]
    strike, kind, iv = float(c["strike"]), c["type"], float(c["iv"])
    o_entry = float(c["mid"])
    if o_entry <= 0:
        return None

    if pdir > 0:
        pull_low = float(d["low"].tail(3).min())
        u_stop = max(min(pull_low - 0.0009 * u_entry, u_entry - 0.8 * atr),
                     u_entry - 1.3 * atr)
        u_t1 = u_entry + 1.0 * atr
        swing_hi = float(d["high"].tail(20).max())
        u_run = min(max(swing_hi, u_entry + 1.8 * atr), u_entry + 2.8 * atr)
    else:
        pull_hi = float(d["high"].tail(3).max())
        u_stop = min(max(pull_hi + 0.0009 * u_entry, u_entry + 0.8 * atr),
                     u_entry + 1.3 * atr)
        u_t1 = u_entry - 1.0 * atr
        swing_lo = float(d["low"].tail(20).min())
        u_run = max(min(swing_lo, u_entry - 1.8 * atr), u_entry - 2.8 * atr)

    t_red = _t_years(dte, min(2, max(1, dte - 1)))
    o_stop = 0.50 * o_entry
    o_t1 = max(float(bs.price(u_t1, strike, t_red, iv * 0.93, R, Q, kind)), 1.35 * o_entry)
    o_run = max(float(bs.price(u_run, strike, t_red, iv * 0.88, R, Q, kind)), 1.8 * o_entry)

    return types.SimpleNamespace(
        session_date=str(session_date), direction=pdir, expiry=str(exp_date),
        dte=dte, strike=strike, o_entry=o_entry, o_stop=o_stop,
        u_stop=u_stop, u_t1=u_t1, u_run=u_run,
        o_t1_pct=o_t1 / o_entry - 1.0, o_run_pct=o_run / o_entry - 1.0,
    )


def summarize(df: pd.DataFrame, label: str):
    if df.empty:
        print(f"{label}: no qualified setups")
        return
    wins = df[df["pnl_pct"] > 0]
    losses = df[df["pnl_pct"] <= 0]
    gains, loss = wins["pnl_pct"].sum(), -losses["pnl_pct"].sum()
    pf = gains / loss if loss > 0 else np.inf
    print(f"{label}: {len(df)} setups ({df.date.min()} -> {df.date.max()})")
    print(f"  win rate {len(wins)/len(df):.0%}  ·  avg {df['pnl_pct'].mean():+.1%} "
          f"of premium ({df['R'].mean():+.2f} R)  ·  median {df['pnl_pct'].median():+.1%}  ·  "
          f"profit factor {pf:.2f}  ·  ${df['pnl_$'].mean():+.0f}/trade avg (1 contract)")
    print("  " + df["label"].value_counts().to_string().replace("\n", "  ·  "))


def main():
    print("Refreshing SPY/VIX history...")
    daily = loader.load_history(start="2018-01-01", refresh=True)
    daily.index = pd.DatetimeIndex(daily.index)

    pb = dsig.pullback(daily)
    pb.index = pd.DatetimeIndex(pb.index)
    dates = pb[pb["direction"] != 0].index
    dates = dates[dates >= daily.index[210]]   # past signal warmup

    rows = []
    for ts in dates:
        plan = build_daily_plan(daily, ts.date())
        if plan is None:
            continue
        label, pnl, run_hit = outcome(plan, daily)
        if label == "no-data":
            continue
        rows.append({"date": ts.date(), "dir": plan.direction, "dte": plan.dte,
                     "o_entry": plan.o_entry, "label": label, "pnl_pct": pnl,
                     "R": pnl / 0.50, "pnl_$": pnl * plan.o_entry * 100})

    df = pd.DataFrame(rows)
    if df.empty:
        print("no qualified plans in the sample")
        return
    df["date"] = pd.to_datetime(df["date"])

    print(f"\n{'='*100}\nFULL HISTORY (2018-2026, daily-open entry -- approximation of the "
          f"real intraday plan)\n{'='*100}")
    summarize(df, "All years")
    print()
    for yr, grp in df.groupby(df["date"].dt.year):
        summarize(grp, f"  {yr}")

    print(f"\n{'='*100}\nTRAIN (2018-2023) vs TEST (2024-2026, the same window the original "
          f"intraday backtest covered)\n{'='*100}")
    summarize(df[df["date"] < SPLIT], "Train 2018-2023")
    summarize(df[df["date"] >= SPLIT], "Test  2024-2026")


if __name__ == "__main__":
    main()
