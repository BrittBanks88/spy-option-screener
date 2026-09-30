"""Out-of-sample check on backtest_1000_account.py's "best combo per ticker".

Picking the single best-performing config out of 24 tested (2 target deltas x
2 profit targets x 6 signals) using the FULL history and then reporting that
config's full-history numbers is hindsight bias -- of course the best of 24
looks good over the exact period it was chosen on. This script asks the
honest question instead:

    1. Pick the best config using only TRAIN data (2018-01-01 -> 2023-12-31).
    2. Report how that SAME config actually did on TEST data (2024-01-01 ->
       today) it never saw when it was chosen.

If train and test performance are similar, the edge is probably real. If test
is much worse than train (or negative), the "best combo" was mostly noise.

    python research/backtest_walkforward.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spy_option_screener.data import loader                             # noqa: E402
from spy_option_screener.signals import strategies as st_mod            # noqa: E402
from spy_option_screener.backtest import engine, metrics                # noqa: E402
from spy_option_screener.backtest.rules import TradeRules               # noqa: E402
from research.backtest_1000_account import (                            # noqa: E402
    load_generic, SIGNALS, TARGET_DELTAS, PROFIT_TARGETS, START_CAPITAL,
    HIST_START,
)

SPLIT = pd.Timestamp("2024-01-01")


def run_full(df, sig_name, target_delta, profit_target):
    sig = (st_mod.combine(df) if sig_name == "combo" else st_mod.STRATEGIES[sig_name](df))
    rules = TradeRules(
        target_dte=21, max_dte=28, time_stop_dte=3,
        target_delta=target_delta, min_confidence=0.15,
        fixed_contracts=1, profit_target_pct=profit_target,
        stop_loss_pct=0.50, exit_on_signal_flip=False,
    )
    return engine.run(df, sig, rules, starting_capital=START_CAPITAL)


def period_stats(equity, trades, start=None, end=None):
    """Re-derive stats for a date sub-window from a full-history run, treating
    the sub-window's starting equity as its own $1,000-equivalent base."""
    e = equity.copy()
    if start is not None:
        e = e[e.index >= start]
    if end is not None:
        e = e[e.index <= end]
    if e.empty:
        return {"n_trades": 0, "return_pct": np.nan, "avg_monthly_pnl": np.nan}
    t = trades[(trades["entry_date"] >= (start or trades["entry_date"].min())) &
               (trades["entry_date"] <= (end or trades["entry_date"].max()))] \
        if not trades.empty else trades
    base = e.iloc[0]
    ret = e.iloc[-1] / base - 1.0 if base else np.nan
    months = e.resample("ME").last().dropna()
    mp = months.diff().fillna(months.iloc[0] - base)
    return {
        "n_trades": len(t), "return_pct": ret,
        "win_rate": (t["pnl"] > 0).mean() if len(t) else np.nan,
        "avg_monthly_pnl": mp.mean(), "median_monthly_pnl": mp.median(),
        "max_drawdown": (e / e.cummax() - 1.0).min(),
        "months_ge_1500": int((mp >= 1500).sum()), "n_months": len(mp),
    }


def main():
    tickers = sys.argv[1:] or ["SPY", "NVDA", "TSLA", "PLTR"]
    rows = []
    for ticker in tickers:
        print(f"Loading {ticker}...")
        df = (loader.load_history(start=HIST_START) if ticker == "SPY"
              else load_generic(ticker, refresh=False))
        train_df = df[df.index < SPLIT]
        if len(train_df) < 300:
            print(f"  skipping {ticker}: not enough pre-2024 history "
                  f"({len(train_df)} sessions)")
            continue

        # 1. pick the best config using ONLY the training window
        best = None
        for sig_name, tdelta, ptgt in product(SIGNALS, TARGET_DELTAS, PROFIT_TARGETS):
            equity, trades, _ = run_full(train_df, sig_name, tdelta, ptgt)
            train_stats = period_stats(equity, trades)
            score = train_stats["avg_monthly_pnl"]
            if score is not None and (best is None or (pd.notna(score) and score > best[0])):
                best = (score, sig_name, tdelta, ptgt, train_stats)
        if best is None:
            continue
        _, sig_name, tdelta, ptgt, train_stats = best

        # 2. run that SAME config over the FULL series, then split the
        #    resulting equity/trades into the train and test windows.
        equity, trades, _ = run_full(df, sig_name, tdelta, ptgt)
        train_eval = period_stats(equity, trades, end=SPLIT - pd.Timedelta(days=1))
        test_eval = period_stats(equity, trades, start=SPLIT)

        rows.append({"ticker": ticker, "signal": sig_name, "target_delta": tdelta,
                     "profit_target": ptgt,
                     "train_avg_monthly": train_eval["avg_monthly_pnl"],
                     "train_return": train_eval["return_pct"],
                     "train_trades": train_eval["n_trades"],
                     "train_months_ge_1500": train_eval["months_ge_1500"],
                     "train_n_months": train_eval["n_months"],
                     "test_avg_monthly": test_eval["avg_monthly_pnl"],
                     "test_return": test_eval["return_pct"],
                     "test_trades": test_eval["n_trades"],
                     "test_max_dd": test_eval["max_drawdown"],
                     "test_months_ge_1500": test_eval["months_ge_1500"],
                     "test_n_months": test_eval["n_months"]})

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 170)
    print(f"\n{'='*170}\nConfig chosen using ONLY 2018-2023 data, then evaluated on "
          f"2024-2026 (never seen when picked):\n{'='*170}")
    show = res.copy()
    for c in ["train_return", "test_return", "test_max_dd"]:
        show[c] = show[c].map(lambda x: f"{x:.0%}" if pd.notna(x) else "-")
    for c in ["train_avg_monthly", "test_avg_monthly"]:
        show[c] = show[c].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "-")
    print(show.to_string(index=False))
    print("\nIf test_avg_monthly is close to (or better than) train_avg_monthly, "
          "the edge likely generalizes. If test is much worse or negative, the "
          "'best combo' picked on the earlier run was mostly overfit noise.")


if __name__ == "__main__":
    main()
