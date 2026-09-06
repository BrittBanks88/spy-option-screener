"""Does buying an option to hold ONE night make money? (Short answer: no.)

Buys a call at the close on day t, marks it at day t+1's open, Black-Scholes on
a VIX-derived surface with a slippage haircut each side. Runs several filters
and strike depths. Compare with the pure equity overnight drift (~+3 bps).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spy_option_screener.data import loader                           # noqa: E402
from spy_option_screener.pricing import black_scholes as bs           # noqa: E402
from spy_option_screener.signals import indicators as ind             # noqa: E402

R, Q, SLIP = 0.04, 0.013, 0.01


def sim(d, mask, delta_t=0.65, dte0=2, label=""):
    out = []
    for i in range(len(d) - 1):
        if not mask.iloc[i]:
            continue
        S = d["close"].iloc[i]
        iv = max(d["vix"].iloc[i] / 100 * 0.95, 0.06)
        Ks = np.arange(round(S) - 18, round(S) + 18)
        dl = bs.greeks(S, Ks, dte0 / 365, iv, R, Q, "call")["delta"]
        K = Ks[int(np.argmin(np.abs(dl - delta_t)))]
        entry = float(bs.price(S, K, dte0 / 365, iv, R, Q, "call")) * (1 + SLIP)
        Sn = d["open"].iloc[i + 1]
        exit_ = float(bs.price(Sn, K, (dte0 - 1) / 365, iv, R, Q, "call")) * (1 - SLIP)
        out.append(exit_ / entry - 1.0)
    x = pd.Series(out)
    if len(x) < 30:
        print(f"{label:44} n={len(x)}  too few")
        return
    pf = x[x > 0].sum() / max(-x[x <= 0].sum(), 1e-9)
    print(f"{label:44} n={len(x):4d}  avg {x.mean():+.1%}  median {x.median():+.1%}"
          f"  win {(x > 0).mean():.0%}  PF {pf:.2f}")


def main():
    d = loader.load_history(start="2012-01-01")
    d["ma50"] = ind.sma(d["close"], 50)
    d["day_ret"] = d["close"] / d["open"] - 1.0
    d["on_equity"] = d["open"].shift(-1) / d["close"] - 1.0
    d = d.dropna()

    print("Equity overnight drift (close→open), all days:")
    print(f"  avg {d['on_equity'].mean()*1e4:+.1f} bps   up {(d['on_equity']>0).mean():.0%}"
          f"   std {d['on_equity'].std()*1e4:.0f} bps\n")

    print("Option held one night (call, entry at close, exit at next open):")
    allmask = pd.Series(True, index=d.index)
    sim(d, allmask, 0.65, 2, "every night · 0.65Δ · 2 DTE")
    sim(d, allmask, 0.80, 2, "every night · 0.80Δ (deep ITM) · 2 DTE")
    sim(d, d["day_ret"] < -0.005, 0.65, 2, "down day >0.5% · 0.65Δ")
    sim(d, (d["day_ret"] < -0.008) & (d["close"] > d["ma50"]), 0.80, 3,
        "down >0.8% in uptrend · 0.80Δ · 3 DTE  (best found)")
    sim(d, d["day_ret"] > 0.005, 0.65, 2, "up day >0.5% · 0.65Δ")

    print("\nConclusion: the ~2% option round-trip cost swamps the ~3-5 bps of "
          "overnight index drift. Hold overnight only as part of a position that "
          "already has an edge (e.g. the pullback plan) — not as a scalp.")


if __name__ == "__main__":
    main()
