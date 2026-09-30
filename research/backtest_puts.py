"""Hunt for a real downtrend/PUT edge, tested honestly.

The live pullback->continuation plan mirrors its call logic for puts, and
that mirror has no track record (9 historical trades, 11% win rate) --
almost certainly because SPY has barely spent any time in a real downtrend
since 2019. This script tests ALL SIX existing daily signals on their PUT
side only, using the same weekly-option mechanics already validated for
calls (ATR-based stop/target/runner, nearest 3-9 DTE weekly, reconstructed
BS pricing), and reports each one specifically against:

  - the full history
  - ONLY the down years (2018, 2022 -- the two years SPY actually fell)
  - a train/test split, for whatever that's worth on a small sample

Be upfront going in: real, sustained SPY downtrends are rare in this data
(basically 2018 Q4, the 2020 crash, and 2022). Whatever comes out of this
is tested on a much thinner sample than the call side ever was -- treat any
"edge" found here with proportionally more suspicion, not less.

    python research/backtest_puts.py
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
from spy_option_screener.signals import strategies as st_mod            # noqa: E402
from spy_option_screener.signals import indicators as ind               # noqa: E402
from spy_option_screener.screener import chain as chain_mod             # noqa: E402
from spy_option_screener.screener import score as score_mod             # noqa: E402
from spy_option_screener.screener.trade_plan import _t_years, R, Q      # noqa: E402
from research.backtest_trade_plan import outcome                        # noqa: E402

BUY_ZONE = (0.35, 0.60)
MIN_CONF = 0.15
DOWN_YEARS = {2018, 2022}
SPLIT = pd.Timestamp("2024-01-01")


def build_put_plan(daily, session_date, atr14):
    """Same level/pricing mechanics as the validated call plan (ATR stop/t1/
    runner, nearest weekly 3-9 DTE, reconstructed BS pricing) -- direction
    fixed to PUT, no trend-confirmation gate (each signal brings its own)."""
    d = daily.loc[:pd.Timestamp(session_date)]
    close = d["close"]
    entry_date = mcal.next_trading_day(session_date)
    entry_ts = pd.Timestamp(entry_date)
    if entry_ts not in daily.index:
        return None
    u_entry = float(daily.loc[entry_ts, "open"])
    vix = float(d["vix"].iloc[-1])
    rv = float(d["realized_vol_20"].iloc[-1])
    atr = float(atr14.loc[session_date]) if session_date in atr14.index else np.nan
    if not np.isfinite(atr) or atr <= 0:
        return None

    exp_date, dte = mcal.next_weekly_expiry(entry_date, min_dte=3, max_dte=9)
    chain = chain_mod.synthetic_chain(u_entry, vix, dte,
                                      vix_baseline=float(d["vix"].tail(40).mean()),
                                      r=R, q=Q)
    scored = score_mod.score_chain(chain, -1, u_entry, rv, confidence=1.0)
    lo, hi = BUY_ZONE
    band = scored[scored["delta"].abs().between(lo, hi)]
    if band.empty and scored.empty:
        return None
    c = (band if not band.empty else scored).iloc[0]
    strike, kind, iv = float(c["strike"]), c["type"], float(c["iv"])
    o_entry = float(c["mid"])
    if o_entry <= 0:
        return None

    pull_hi = float(d["high"].tail(3).max())
    u_stop = min(max(pull_hi + 0.0009 * u_entry, u_entry + 0.8 * atr), u_entry + 1.3 * atr)
    u_t1 = u_entry - 1.0 * atr
    swing_lo = float(d["low"].tail(20).min())
    u_run = max(min(swing_lo, u_entry - 1.8 * atr), u_entry - 2.8 * atr)

    t_red = _t_years(dte, min(2, max(1, dte - 1)))
    o_t1 = max(float(bs.price(u_t1, strike, t_red, iv * 0.93, R, Q, kind)), 1.35 * o_entry)
    o_run = max(float(bs.price(u_run, strike, t_red, iv * 0.88, R, Q, kind)), 1.8 * o_entry)

    return types.SimpleNamespace(
        session_date=str(session_date), direction=-1, expiry=str(exp_date),
        dte=dte, strike=strike, o_entry=o_entry, o_stop=0.5 * o_entry,
        u_stop=u_stop, u_t1=u_t1, u_run=u_run,
        o_t1_pct=o_t1 / o_entry - 1.0, o_run_pct=o_run / o_entry - 1.0,
    )


def run_signal(daily, sig_name, atr14):
    sig = (st_mod.combine(daily) if sig_name == "combo" else st_mod.STRATEGIES[sig_name](daily))
    puts = sig[(sig["direction"] < 0) & (sig["confidence"] >= MIN_CONF)]
    puts = puts[puts.index >= daily.index[210]]
    rows = []
    for ts in puts.index:
        plan = build_put_plan(daily, ts.date(), atr14)
        if plan is None:
            continue
        label, pnl, run_hit = outcome(plan, daily)
        if label == "no-data":
            continue
        rows.append({"date": ts.date(), "label": label, "pnl_pct": pnl,
                     "R": pnl / 0.50, "pnl_$": pnl * plan.o_entry * 100})
    return pd.DataFrame(rows)


def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "win_rate": np.nan, "avg_pct": np.nan, "pf": np.nan,
                "avg_$": np.nan}
    wins, losses = df[df.pnl_pct > 0], df[df.pnl_pct <= 0]
    gains, loss = wins.pnl_pct.sum(), -losses.pnl_pct.sum()
    return {"n": len(df), "win_rate": len(wins) / len(df), "avg_pct": df.pnl_pct.mean(),
            "pf": gains / loss if loss > 0 else np.inf, "avg_$": df["pnl_$"].mean()}


def fmt(s: dict) -> str:
    if s["n"] == 0:
        return "n=0"
    pf = "inf" if not np.isfinite(s["pf"]) else f"{s['pf']:.2f}"
    return (f"n={s['n']:<3d} win={s['win_rate']:.0%}  avg={s['avg_pct']:+.1%}  "
            f"PF={pf}  ${s['avg_$']:+.0f}/trade")


def main():
    print("Refreshing SPY/VIX history...")
    daily = loader.load_history(start="2018-01-01", refresh=True)
    daily.index = pd.DatetimeIndex(daily.index)
    atr14 = ind.atr(daily["high"], daily["low"], daily["close"], 14)
    atr14.index = pd.DatetimeIndex(atr14.index).date
    atr14 = pd.Series(atr14.values, index=pd.Index(atr14.index))

    names = list(st_mod.STRATEGIES) + ["combo"]
    print(f"\n{'='*100}\nPUT-only performance per signal (weekly options, same mechanics "
          f"already validated for calls)\n{'='*100}")
    print(f"{'signal':<20}{'ALL YEARS':<32}{'DOWN YEARS (2018,2022)':<32}{'2024-2026'}")

    results = {}
    for name in names:
        df = run_signal(daily, name, atr14)
        results[name] = df
        df_dt = df.copy()
        if not df_dt.empty:
            df_dt["date"] = pd.to_datetime(df_dt["date"])
        down = df_dt[df_dt["date"].dt.year.isin(DOWN_YEARS)] if not df_dt.empty else df_dt
        recent = df_dt[df_dt["date"] >= SPLIT] if not df_dt.empty else df_dt
        print(f"{name:<20}{fmt(stats(df)):<32}{fmt(stats(down)):<32}{fmt(stats(recent))}")

    print(f"\n{'='*100}\nNote: real SPY downtrend data is thin -- 2018 and 2022 are the only "
          f"two calendar years SPY actually finished down since 2018. Any 'edge' above is "
          f"tested on a much smaller, noisier sample than the call side. Reconstructed "
          f"option prices. Not investment advice.\n{'='*100}")


if __name__ == "__main__":
    main()
