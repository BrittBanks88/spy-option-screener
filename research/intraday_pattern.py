"""Research: how does SPY behave from the 9:30 open through the 4:00 close?

Pulls what free data allows:
  * 1-hour bars, ~2 years  (yfinance cap)   -> intraday shape, vol, volume
  * 30-min bars, ~60 days                    -> finer 'recent regime' view
  * daily OHLC, 2005->now (cached)           -> overnight vs intraday, gaps

Writes a text report to outputs/intraday_report.txt and charts to outputs/.
Nothing here is a trade recommendation -- it's a description of history.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)
ET = "America/New_York"
REG_OPEN, REG_CLOSE = pd.Timestamp("09:30").time(), pd.Timestamp("16:00").time()


def _flat(d):
    if isinstance(d.columns, pd.MultiIndex):
        d = d.copy()
        d.columns = d.columns.get_level_values(0)
    return d


def get_intraday(interval, period):
    import yfinance as yf
    d = _flat(yf.download("SPY", interval=interval, period=period,
                          progress=False, auto_adjust=False))
    d = d.tz_convert(ET) if d.index.tz else d.tz_localize(ET)
    d = d[(d.index.time >= REG_OPEN) & (d.index.time < REG_CLOSE)]
    d["date"] = d.index.date
    d["bucket"] = d.index.strftime("%H:%M")
    d["ret"] = d.groupby("date")["Close"].pct_change()
    # first bucket return = open->close of that bar
    first = d.groupby("date").head(1).index
    d.loc[first, "ret"] = (d.loc[first, "Close"] / d.loc[first, "Open"] - 1.0)
    return d


def intraday_shape(d, label, lines):
    g = d.groupby("bucket")
    tab = pd.DataFrame({
        "mean_ret_bps": g["ret"].mean() * 1e4,
        "median_bps": g["ret"].median() * 1e4,
        "stdev_bps": g["ret"].std() * 1e4,
        "mean_abs_bps": g["ret"].apply(lambda s: s.abs().mean()) * 1e4,
        "share_up": g["ret"].apply(lambda s: (s > 0).mean()),
        "avg_volume_M": g["Volume"].mean() / 1e6,
    })
    # cumulative average path from the open
    cum = d.pivot_table(index="date", columns="bucket", values="ret").fillna(0)
    cum = (1 + cum).cumprod(axis=1).mean(axis=0) - 1.0
    tab["cum_from_open_bps"] = cum * 1e4

    lines.append(f"\n=== Intraday shape by time bucket ({label}) ===")
    lines.append(f"days: {d['date'].nunique()}   "
                 f"{d.index.min().date()} -> {d.index.max().date()}")
    lines.append(tab.round(2).to_string())
    return tab


def open_vs_rest(d, label, lines):
    piv = d.pivot_table(index="date", columns="bucket", values="ret")
    b = list(piv.columns)
    first, rest = piv[b[0]], (1 + piv[b[1:]]).prod(axis=1) - 1.0
    df = pd.DataFrame({"first": first, "rest": rest}).dropna()
    up = df[df["first"] > 0]
    dn = df[df["first"] < 0]
    corr = df["first"].corr(df["rest"])
    lines.append(f"\n=== First bar vs rest of day ({label}) ===")
    lines.append(f"corr(first bar, rest of day) = {corr:+.3f}")
    lines.append(f"after an UP first bar  (n={len(up)}): rest-of-day avg "
                 f"{up['rest'].mean()*1e4:+.1f} bps, share up {(up['rest']>0).mean():.0%}")
    lines.append(f"after a  DOWN first bar (n={len(dn)}): rest-of-day avg "
                 f"{dn['rest'].mean()*1e4:+.1f} bps, share up {(dn['rest']>0).mean():.0%}")


def overnight_vs_intraday(daily, lines):
    df = daily.copy()
    df["overnight"] = df["open"] / df["close"].shift(1) - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    df = df.dropna(subset=["overnight", "intraday"])

    def blk(sub, name):
        on, idr = sub["overnight"], sub["intraday"]
        gON = (1 + on).prod() - 1.0
        gID = (1 + idr).prod() - 1.0
        lines.append(f"  {name:<11} n={len(sub):>5}  "
                     f"overnight: total {gON*100:>8.1f}%  avg {on.mean()*1e4:>6.2f}bps  up {(on>0).mean():.0%}   | "
                     f"intraday: total {gID*100:>8.1f}%  avg {idr.mean()*1e4:>6.2f}bps  up {(idr>0).mean():.0%}")

    lines.append("\n=== Overnight (prev close -> open) vs Intraday (open -> close) ===")
    blk(df, "2005-now")
    for y0, y1 in [(2005, 2010), (2010, 2015), (2015, 2020), (2020, 2026)]:
        sub = df[(df.index.year >= y0) & (df.index.year < y1)]
        if len(sub):
            blk(sub, f"{y0}-{y1-1}")
    sub = df[df.index.year >= 2024]
    blk(sub, "2024-now")


def gap_behavior(daily, lines):
    df = daily.copy()
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    df = df.dropna()
    q = df["gap"].quantile([.1, .25, .75, .9])
    lines.append("\n=== Gap fade / follow (open->close conditioned on the gap) ===")
    for name, mask in {
        "big gap DOWN (<p10)": df["gap"] < q[.10],
        "small gap down": (df["gap"] >= q[.10]) & (df["gap"] < 0),
        "small gap up": (df["gap"] >= 0) & (df["gap"] <= q[.90]),
        "big gap UP (>p90)": df["gap"] > q[.90],
    }.items():
        s = df.loc[mask, "intraday"]
        lines.append(f"  {name:<22} n={len(s):>5}  intraday avg {s.mean()*1e4:+7.2f} bps  "
                     f"up {(s>0).mean():.0%}   (fade = opposite sign to gap)")


def day_of_week(daily, lines):
    df = daily.copy()
    df["intraday"] = df["close"] / df["open"] - 1.0
    df["overnight"] = df["open"] / df["close"].shift(1) - 1.0
    df["dow"] = df.index.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    g = df.groupby("dow")
    tab = pd.DataFrame({
        "intraday_avg_bps": g["intraday"].mean() * 1e4,
        "intraday_up": g["intraday"].apply(lambda s: (s > 0).mean()),
        "overnight_avg_bps": g["overnight"].mean() * 1e4,
    }).reindex(order)
    lines.append("\n=== Day-of-week (2005-now) ===")
    lines.append(tab.round(2).to_string())


def charts(h1, tab_h1, daily):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    b = tab_h1.index
    ax[0].plot(b, tab_h1["cum_from_open_bps"], marker="o")
    ax[0].axhline(0, color="k", lw=.6)
    ax[0].set_title("Avg cumulative SPY move from 9:30 open (bps)\n1h bars, ~2yr")
    ax[0].set_ylabel("bps"); ax[0].tick_params(axis="x", rotation=45)

    ax[1].bar(b, tab_h1["mean_abs_bps"])
    ax[1].set_title("Avg absolute move per hour (volatility, bps)")
    ax[1].tick_params(axis="x", rotation=45)

    ax[2].bar(b, tab_h1["avg_volume_M"])
    ax[2].set_title("Avg volume per hour (millions of shares)")
    ax[2].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "intraday_shape.png", dpi=110)

    # overnight vs intraday cumulative growth of $1
    df = daily.copy()
    df["overnight"] = df["open"] / df["close"].shift(1) - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    df = df.dropna()
    fig2, a2 = plt.subplots(figsize=(10, 5))
    a2.plot(df.index, (1 + df["overnight"]).cumprod(), label="overnight only (close->open)")
    a2.plot(df.index, (1 + df["intraday"]).cumprod(), label="intraday only (open->close)")
    a2.plot(df.index, df["close"] / df["close"].iloc[0], label="buy & hold", color="k", lw=.8)
    a2.set_yscale("log"); a2.legend(); a2.set_title("SPY: $1 compounded, overnight vs intraday (2005-now)")
    fig2.tight_layout()
    fig2.savefig(OUT / "overnight_vs_intraday.png", dpi=110)
    print(f"charts -> {OUT}/intraday_shape.png, {OUT}/overnight_vs_intraday.png")


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spy_option_screener.data import loader

    lines = ["SPY INTRADAY PATTERN RESEARCH", "=" * 60]

    h1 = get_intraday("1h", "730d")
    tab_h1 = intraday_shape(h1, "1-hour bars, ~2 years", lines)
    open_vs_rest(h1, "1h, ~2yr", lines)

    m30 = get_intraday("30m", "60d")
    intraday_shape(m30, "30-min bars, last 60 days", lines)
    open_vs_rest(m30, "30m, 60d", lines)

    daily = loader.load_history(start="2005-01-01")
    overnight_vs_intraday(daily, lines)
    gap_behavior(daily, lines)
    day_of_week(daily, lines)

    report = "\n".join(lines)
    (OUT / "intraday_report.txt").write_text(report)
    print(report)
    charts(h1, tab_h1, daily)


if __name__ == "__main__":
    main()
