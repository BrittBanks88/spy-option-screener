"""CLI: the single best SPY weekly contract to buy for the current/next session.

    python daily_pick.py                 # 30-min bars, decide as of 10:30 ET
    python daily_pick.py --entry 10:00 --interval 30m --refresh

Combines the gap-continuation + opening-range-breakout intraday signals with
the daily trend/mean-reversion bias, then ranks the nearest weekly chain.
Price is not a filter. Not investment advice.
"""
from __future__ import annotations

import argparse

from spy_option_screener.screener.daily_pick import todays_pick


def _fmt_contract(c: dict) -> str:
    if not c:
        return "  (none in the directional-buy zone)"
    return (f"  {c['type'].upper():4} ${c['strike']:.0f}   "
            f"mid ${c['mid']:.2f}  (${c['mid']*100:.0f}/ct)   "
            f"Δ {c['delta']:+.2f}   θ -${abs(c['theta'])*100:.0f}/day   "
            f"IV {c['iv']:.1%}\n"
            f"        breakeven ${c['breakeven']:.2f} ({c['be_pct']:+.2%})   "
            f"chance of profit {c['chance_of_profit']:.0%}   "
            f"leverage {c['leverage']:.0f}x   score {c['score']:.0f}/100")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="30m", choices=["1h", "30m", "15m"])
    ap.add_argument("--entry", default="10:30", help="decision/entry time ET")
    ap.add_argument("--all-strikes", action="store_true",
                    help="don't restrict to the 0.20-0.62 delta buy zone")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    p = todays_pick(interval=args.interval, entry_time=args.entry,
                    buy_zone=not args.all_strikes, refresh=args.refresh)

    print("=" * 74)
    print(f"SPY DAILY PICK   as of {p.asof}")
    print(f"  {p.day_note}")
    print("-" * 74)
    print(f"signal source: {p.session_date} session   ·   "
          f"spot ${p.spot:.2f}   VIX {p.vix:.1f}   "
          f"realized vol {p.realized_vol:.1%}   gap {p.gap:+.2%}")
    print("=" * 74)
    print("\nSignals:")
    for r in p.reasons:
        print(f"  • {r}")
    verdict = {1: "CALLS", -1: "PUTS", 0: "STAND ASIDE"}[p.direction]
    print(f"\n  => direction: {verdict}    conviction: {p.conviction:.0%}")

    print("\n" + "-" * 74)
    print(p.headline())
    print("-" * 74)

    print(f"\nBest CALL (exp {p.expiry}, {p.dte}d):")
    print(_fmt_contract(p.call_alt))
    print(f"\nBest PUT  (exp {p.expiry}, {p.dte}d):")
    print(_fmt_contract(p.put_alt))
    print(f"\n{p.note}")


if __name__ == "__main__":
    main()
