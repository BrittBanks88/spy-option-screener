"""Backtest the actual goal across multiple underlyings: $1,000 account, one
option contract at a time, held ~3 weeks, across every daily signal -- over
the full cached history. Answers one question honestly per ticker: what would
this really have made or lost, month by month, on $1,000?

    python research/backtest_1000_account.py                # SPY, NVDA, TSLA, PLTR
    python research/backtest_1000_account.py AAPL MSFT       # custom tickers

SPY uses the real VIX (pricing/vol_model.py). Every other ticker has no free
live implied-vol history, so its "vix" column is a PROXY: trailing 20-day
realized volatility x 1.15 (a stand-in volatility-risk-premium multiplier --
real option markets price single stocks at a real premium to their own
realized vol too, this just isn't fit to any actual quote). That means
non-SPY results are a rougher approximation than the SPY ones and should be
read as "does this signal/underlying combo look directionally promising",
not as a return you could bank on. Not investment advice.
"""
from __future__ import annotations

import sys
import datetime as dt
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spy_option_screener.data import loader                             # noqa: E402
from spy_option_screener.signals import strategies as st_mod            # noqa: E402
from spy_option_screener.backtest import engine, metrics                # noqa: E402
from spy_option_screener.backtest.rules import TradeRules               # noqa: E402

START_CAPITAL = 1_000.0
SIGNALS = list(st_mod.STRATEGIES) + ["combo"]
TARGET_DELTAS = [0.35, 0.45]
PROFIT_TARGETS = [0.50, 1.00]
IV_PREMIUM = 1.15          # realized-vol x this = the IV proxy for non-SPY names
HIST_START = "2018-01-01"


def load_generic(ticker: str, start=HIST_START, refresh=False) -> pd.DataFrame:
    """Same shape as loader.load_history, but for any yfinance ticker, with a
    realized-vol-based IV proxy standing in for a real vol index."""
    import yfinance as yf
    cache = Path(__file__).resolve().parents[1] / "cache" / f"{ticker}_daily.parquet"
    cache.parent.mkdir(exist_ok=True)
    if refresh or not cache.exists():
        raw = yf.download(ticker, start=start, end=dt.date.today().isoformat(),
                          auto_adjust=False, progress=False)
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.copy()
            raw.columns = raw.columns.get_level_values(0)
        raw.to_parquet(cache)
    raw = pd.read_parquet(cache)
    df = pd.DataFrame(index=raw.index)
    df["open"], df["high"], df["low"] = raw["Open"], raw["High"], raw["Low"]
    df["close"], df["volume"] = raw["Close"], raw["Volume"]
    df["ret"] = df["close"].pct_change()
    df["realized_vol_20"] = df["ret"].rolling(20).std() * np.sqrt(252.0)
    df["vix"] = (df["realized_vol_20"] * 100.0 * IV_PREMIUM).bfill()
    return df.loc[str(start):].dropna(subset=["close"])


def monthly_pnl(equity: pd.Series) -> pd.Series:
    m = equity.resample("ME").last().dropna()
    return m.diff().fillna(m.iloc[0] - START_CAPITAL)


def run_one(df, sig_name, target_delta, profit_target):
    sig = (st_mod.combine(df) if sig_name == "combo" else st_mod.STRATEGIES[sig_name](df))
    rules = TradeRules(
        target_dte=21, max_dte=28, time_stop_dte=3,
        target_delta=target_delta, min_confidence=0.15,
        fixed_contracts=1, profit_target_pct=profit_target,
        stop_loss_pct=0.50, exit_on_signal_flip=False,
    )
    equity, trades, _ = engine.run(df, sig, rules, starting_capital=START_CAPITAL)
    summ = metrics.summary(equity, trades)
    mp = monthly_pnl(equity)
    summ.update({
        "signal": sig_name, "target_delta": target_delta,
        "profit_target": profit_target,
        "end_equity": equity.iloc[-1] if len(equity) else np.nan,
        "avg_monthly_pnl": mp.mean(), "median_monthly_pnl": mp.median(),
        "pct_months_positive": (mp > 0).mean() if len(mp) else np.nan,
        "n_months": len(mp), "worst_month": mp.min() if len(mp) else np.nan,
        "best_month": mp.max() if len(mp) else np.nan,
    })
    return summ, equity, trades


def fmt_table(res: pd.DataFrame) -> pd.DataFrame:
    show = res[["ticker", "signal", "target_delta", "profit_target", "n_trades",
                "win_rate", "profit_factor", "total_return", "max_drawdown",
                "avg_monthly_pnl", "pct_months_positive", "worst_month",
                "best_month", "end_equity"]].copy()
    for c in ["win_rate", "total_return", "max_drawdown", "pct_months_positive"]:
        show[c] = show[c].map(lambda x: f"{x:.0%}" if pd.notna(x) else "-")
    for c in ["avg_monthly_pnl", "worst_month", "best_month", "end_equity"]:
        show[c] = show[c].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "-")
    show["profit_factor"] = show["profit_factor"].map(
        lambda x: f"{x:.2f}" if np.isfinite(x) else "inf")
    return show


def main():
    tickers = sys.argv[1:] or ["SPY", "NVDA", "TSLA", "PLTR"]
    pd.set_option("display.width", 170)

    all_rows, cache = [], {}
    for ticker in tickers:
        print(f"Loading {ticker}...")
        if ticker == "SPY":
            df = loader.load_history(start=HIST_START)
        else:
            df = load_generic(ticker, refresh=False)
        print(f"  {len(df)} sessions, {df.index[0].date()} -> {df.index[-1].date()}, "
              f"trailing 20d realized vol now: {df['realized_vol_20'].iloc[-1]:.0%}")

        for sig_name, tdelta, ptgt in product(SIGNALS, TARGET_DELTAS, PROFIT_TARGETS):
            summ, equity, trades = run_one(df, sig_name, tdelta, ptgt)
            summ["ticker"] = ticker
            cache[(ticker, sig_name, tdelta, ptgt)] = (equity, trades)
            all_rows.append(summ)

    res = pd.DataFrame(all_rows).sort_values("avg_monthly_pnl", ascending=False)

    print(f"\n{'='*170}\nTOP 15 combos overall, ranked by average monthly $ P&L "
          f"($1,000 account, 1 contract, ~3-week hold):\n{'='*170}")
    print(fmt_table(res.head(15)).to_string(index=False))

    print(f"\n{'='*170}\nBest combo PER TICKER:\n{'='*170}")
    best_per_ticker = res.sort_values("avg_monthly_pnl", ascending=False).groupby(
        "ticker", sort=False).head(1)
    _tmp = best_per_ticker.set_index("ticker")
    best_per_ticker = _tmp.loc[[t for t in tickers if t in _tmp.index]].reset_index()
    print(fmt_table(best_per_ticker).to_string(index=False))

    for _, row in best_per_ticker.iterrows():
        key = (row["ticker"], row["signal"], row["target_delta"], row["profit_target"])
        equity, trades = cache[key]
        mp = monthly_pnl(equity)
        n_pos = int((mp >= 1500).sum())
        print(f"\n{'-'*170}\n{row['ticker']}: best combo = {row['signal']} · "
              f"target delta {row['target_delta']} · profit target "
              f"{row['profit_target']:.0%}  ({mp.index[0]:%b %Y} -> {mp.index[-1]:%b %Y}, "
              f"{row['n_trades']:.0f} trades)\n{'-'*170}")
        for d, v in mp.items():
            bar = ("+" if v >= 0 else "-") * min(int(abs(v) / 25), 50)
            print(f"  {d:%Y-%m}   {v:>+9,.0f}   {bar}")
        print(f"\n  Months clearing +$1,500: {n_pos}/{len(mp)} ({n_pos/len(mp):.0%})  ·  "
              f"avg month ${mp.mean():,.0f}  ·  median ${mp.median():,.0f}  ·  "
              f"worst ${mp.min():,.0f}  ·  best ${mp.max():,.0f}  ·  "
              f"ended at ${equity.iloc[-1]:,.0f}")


if __name__ == "__main__":
    main()
