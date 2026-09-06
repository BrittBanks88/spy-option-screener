"""Backtest the pullback→continuation TRADE PLAN (entry / stop / T1 / runner).

For every session in the intraday window where the plan qualifies, we build the
plan, then walk actual SPY daily bars forward from the entry date to expiry and
score the outcome:

    stop first   -> -50% of premium (the plan's stated option stop)
    T1 first     -> take 2/3 at T1; the last 1/3 rides to the runner or, failing
                    that, is marked at expiration
    neither      -> marked at expiration intrinsic

R = 50% of premium (what a stop costs). Results are reported in $ and in R.
~2-year sample, reconstructed option prices -- directional, not gospel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spy_option_screener.data import loader, intraday as it            # noqa: E402
from spy_option_screener.pricing import black_scholes as bs            # noqa: E402
from spy_option_screener.screener.trade_plan import build_plan          # noqa: E402
from spy_option_screener.signals import strategies as st               # noqa: E402


def outcome(plan, daily):
    """Return (label, pnl_pct_on_premium, run_hit)."""
    d = daily.copy()
    d.index = pd.DatetimeIndex(d.index)
    entry = pd.Timestamp(plan.session_date) + pd.Timedelta(days=1)
    exp = pd.Timestamp(plan.expiry)
    path = d[(d.index > pd.Timestamp(plan.session_date)) & (d.index <= exp)]
    if path.empty:
        return "no-data", 0.0, False

    call = plan.direction > 0
    stop, t1, run = plan.u_stop, plan.u_t1, plan.u_run
    hit_t1 = run_hit = False
    for _, bar in path.iterrows():
        lo, hi = bar["low"], bar["high"]
        stopped = (lo <= stop) if call else (hi >= stop)
        reached = (hi >= t1) if call else (lo <= t1)
        if stopped and not hit_t1:
            return "stop", -0.50, False
        if reached:
            hit_t1 = True
        if hit_t1:
            if (hi >= run) if call else (lo <= run):
                run_hit = True
                break

    if not hit_t1:
        # marked at expiration
        S = float(path.iloc[-1]["close"])
        intr = max(S - plan.strike, 0.0) if call else max(plan.strike - S, 0.0)
        return "time", intr / plan.o_entry - 1.0, False

    tail = plan.o_run_pct if run_hit else _mark_at_exp(plan, path)
    pnl = (2 / 3) * plan.o_t1_pct + (1 / 3) * tail
    return ("T1+run" if run_hit else "T1"), pnl, run_hit


def _mark_at_exp(plan, path):
    S = float(path.iloc[-1]["close"])
    call = plan.direction > 0
    intr = max(S - plan.strike, 0.0) if call else max(plan.strike - S, 0.0)
    return intr / plan.o_entry - 1.0


def main():
    daily = loader.load_history(start="2016-01-01")
    daily.index = pd.DatetimeIndex(daily.index)
    bars = it.load_intraday("1h")
    istart = pd.Timestamp(bars["date"].min())

    pb = st.pullback(daily)
    pb.index = pd.DatetimeIndex(pb.index)
    dates = pb[(pb["direction"] != 0) & (pb.index >= istart)].index

    rows = []
    for ts in dates:
        try:
            plan = build_plan(entry_time="10:00", for_date=ts.date())
        except Exception:
            continue
        if not plan.qualified:
            continue
        label, pnl, run_hit = outcome(plan, daily)
        if label == "no-data":
            continue
        rows.append({
            "date": ts.date(), "dir": plan.direction, "dte": plan.dte,
            "o_entry": plan.o_entry, "t1_pct": plan.o_t1_pct,
            "run_pct": plan.o_run_pct, "label": label,
            "pnl_pct": pnl, "R": pnl / 0.50,
            "pnl_$": pnl * plan.o_entry * 100,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("no qualified plans in the sample")
        return
    wins = df[df["pnl_pct"] > 0]
    print(f"Pullback→continuation TRADE-PLAN backtest  ·  {len(df)} setups  "
          f"({df.date.min()} → {df.date.max()})")
    print("-" * 66)
    print(df["label"].value_counts().to_string())
    print("-" * 66)
    print(f"win rate           {len(wins)/len(df):.0%}")
    print(f"avg trade          {df['pnl_pct'].mean():+.1%} of premium   "
          f"({df['R'].mean():+.2f} R)")
    print(f"median trade       {df['pnl_pct'].median():+.1%}")
    print(f"avg win / avg loss {wins['pnl_pct'].mean():+.0%} / "
          f"{df[df['pnl_pct'] <= 0]['pnl_pct'].mean():+.0%}")
    exp_R = df["R"].mean()
    print(f"expectancy         {exp_R:+.2f} R per trade  "
          f"(${df['pnl_$'].mean():+.0f} on a 1-contract position)")
    gains = wins["pnl_pct"].sum()
    losses = -df[df["pnl_pct"] <= 0]["pnl_pct"].sum()
    print(f"profit factor      {gains/losses:.2f}" if losses else "profit factor  inf")
    print(f"runner hit         {df['label'].eq('T1+run').mean():.0%} of all trades")

    out = Path(__file__).resolve().parents[1] / "outputs" / "trade_plan_backtest.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
