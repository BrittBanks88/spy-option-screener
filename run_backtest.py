"""CLI: backtest a long-SPY-weekly-option strategy on reconstructed prices.

    python run_backtest.py --strategy momentum_breakout --start 2015-01-01

Runs are cached offline after the first data download.
"""
from __future__ import annotations

import argparse

import pandas as pd

from spy_option_screener.data import loader
from spy_option_screener.signals import strategies as st
from spy_option_screener.backtest import engine, metrics
from spy_option_screener.backtest.rules import TradeRules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="combo",
                    choices=list(st.STRATEGIES) + ["combo"])
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--capital", type=float, default=25_000)
    ap.add_argument("--target-dte", type=int, default=5)
    ap.add_argument("--target-delta", type=float, default=0.45)
    ap.add_argument("--risk", type=float, default=0.03)
    ap.add_argument("--entry-time", default="close",
                    help="close | 10:00 | 10:30 | 11:00")
    ap.add_argument("--intraday", action="store_true",
                    help="use gap + opening-range-breakout signals (same-day entry)")
    ap.add_argument("--intraday-interval", default="1h", choices=["1h", "30m"])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    et = None if args.entry_time == "close" else args.entry_time
    rules = TradeRules(target_dte=args.target_dte, target_delta=args.target_delta,
                       risk_per_trade=args.risk, entry_time=et)

    if args.intraday:
        from spy_option_screener.backtest import intraday as bi
        et2 = args.entry_time if args.entry_time != "close" else "10:30"
        equity, trades, marks, feat, sig = bi.run_intraday(
            interval=args.intraday_interval, entry_time=et2, rules=rules,
            starting_capital=args.capital, hist_start=args.start,
            refresh=args.refresh)
        df = loader.load_history(start=args.start, end=args.end)
        print(f"Intraday {args.intraday_interval} · entry {et2} · "
              f"{len(feat)} sessions")
        print(f"Signal days: long-call={int((sig['direction']==1).sum())}  "
              f"long-put={int((sig['direction']==-1).sum())}")
    else:
        df = loader.load_history(start=args.start, end=args.end, refresh=args.refresh)
        print(f"Loaded {len(df):,} SPY trading days  "
              f"{df.index[0].date()} -> {df.index[-1].date()}")
        sig = st.combine(df) if args.strategy == "combo" \
            else st.STRATEGIES[args.strategy](df)
        print(f"Signal days: long-call={int((sig['direction']==1).sum())}  "
              f"long-put={int((sig['direction']==-1).sum())}")
        entry_px = None
        if et is not None:
            from spy_option_screener.data import intraday as it
            bars = it.load_intraday("1h", refresh=args.refresh)
            feat = it.daily_features(bars, df, or_minutes=60, signal_cutoff=et,
                                     entry_times=(et, "10:00", "10:30", "11:00"))
            entry_px = feat[f"px_{et.replace(':', '')}"].dropna()
            df = df.reindex(feat.index).dropna(subset=["close", "vix"])
            sig = sig.reindex(df.index)
            entry_px = entry_px.reindex(df.index)
        equity, trades, marks = engine.run(df, sig, rules,
                                           starting_capital=args.capital,
                                           entry_prices=entry_px,
                                           warmup=40 if et else 210)

    summ = metrics.summary(equity, trades)
    print("\n=== Performance ===")
    for k, v in summ.items():
        if isinstance(v, float):
            if "return" in k or "rate" in k or "cagr" in k or "drawdown" in k or "vol" in k:
                print(f"{k:>18}: {v:>10.2%}")
            else:
                print(f"{k:>18}: {v:>10.2f}")
        else:
            print(f"{k:>18}: {v:>10}")

    if not trades.empty:
        print("\n=== Exit reasons ===")
        print(trades["exit_reason"].value_counts().to_string())
        print("\n=== Last 8 trades ===")
        cols = ["entry_date", "exit_date", "type", "strike", "contracts",
                "pnl", "return_pct", "exit_reason"]
        print(trades[cols].tail(8).to_string(index=False))

    out = "outputs"
    import os
    os.makedirs(out, exist_ok=True)
    equity.to_csv(f"{out}/equity_{args.strategy}.csv")
    trades.to_csv(f"{out}/trades_{args.strategy}.csv", index=False)
    print(f"\nSaved -> {out}/equity_{args.strategy}.csv, {out}/trades_{args.strategy}.csv")


if __name__ == "__main__":
    main()
