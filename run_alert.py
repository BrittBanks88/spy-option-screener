"""Scheduled alert runner — posts the SPY pullback→continuation plan to Slack.

Two moments per trading day (see research/intraday_pattern.py for why):

    --mode close    ~3:50-4:15pm ET   "a setup formed on today's close; here's
                    the plan for tomorrow's open" + tonight's overnight-carry read
    --mode morning  ~10:10-10:30am ET  the same plan with the live 10:00 print

    python run_alert.py --mode morning --webhook $SLACK_WEBHOOK_URL
    python run_alert.py --mode auto --dry-run            # pick mode by the clock

By default it posts ONLY when a setup qualifies (so #trade-post isn't spammed
with "nothing today"). Pass --always to post the no-setup card too.

Standalone overnight option buys are NOT alerted — they lose in backtest
(research/backtest_overnight.py). The overnight line rates carrying the
pullback position, nothing more. Not investment advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

from spy_option_screener import _env
from spy_option_screener.data import market_calendar as mcal
from spy_option_screener.screener.trade_plan import build_plan

_env.load()
ET = ZoneInfo("America/New_York")
_MARKERS = Path(__file__).resolve().parent / "outputs"


def _auto_mode(now: dt.datetime) -> str | None:
    t = now.time()
    if dt.time(9, 55) <= t <= dt.time(11, 30):
        return "morning"
    if dt.time(15, 30) <= t <= dt.time(16, 30):
        return "close"
    return None


def _post(webhook: str, payload: dict) -> int:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(webhook, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:      # noqa: S310
        return r.status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "morning", "close"])
    ap.add_argument("--webhook", default=os.environ.get("SLACK_WEBHOOK_URL"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the payload, don't POST")
    ap.add_argument("--always", action="store_true",
                    help="post even when there is no qualifying setup")
    ap.add_argument("--full", action="store_true",
                    help="post the detailed card instead of the simple one")
    ap.add_argument("--force", action="store_true",
                    help="run even outside the usual windows / on a non-trading day")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="ignore the once-per-day-per-mode marker file")
    args = ap.parse_args()

    now = dt.datetime.now(ET)
    if not mcal.is_trading_day(now.date()) and not args.force:
        print(f"{now.date()} is not a trading day — nothing to do.")
        return 0

    mode = args.mode
    if mode == "auto":
        mode = _auto_mode(now)
        if mode is None:
            if not args.force:
                print(f"{now:%H:%M %Z} is outside the alert windows "
                      "(09:55-11:30 or 15:30-16:30 ET). Use --force or --mode.")
                return 0
            mode = "morning"

    marker = _MARKERS / f".alert_{mode}_{now:%Y%m%d}"
    if marker.exists() and not args.no_dedupe:
        print(f"already ran {mode} for {now.date()} ({marker.name}) — skipping.")
        return 0

    interval = "30m" if mode == "morning" else "1h"
    plan = build_plan(entry_time="10:00", interval=interval)

    style = "full" if args.full else "simple"
    payload = plan.slack_payload(style=style)
    if mode == "close" and plan.qualified:
        payload["blocks"].insert(1, {"type": "context", "elements": [
            {"type": "mrkdwn", "text": ":clock4: forms at tomorrow's open — "
             "re-check around 10am ET"}]})

    should_post = plan.qualified or args.always
    print(f"[{now:%Y-%m-%d %H:%M %Z}] mode={mode} style={style} "
          f"qualified={plan.qualified} "
          f"post={should_post and bool(args.webhook) and not args.dry_run}")
    print(plan.slack_simple_text() if not args.full else plan.slack_text())

    if not should_post:
        return 0
    if args.dry_run or not args.webhook:
        if not args.webhook:
            print("\n(no --webhook / SLACK_WEBHOOK_URL — not posting)")
        print("\n--- payload ---")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    status = _post(args.webhook, payload)
    print(f"\nposted to Slack — HTTP {status}")
    if status < 300:
        try:
            _MARKERS.mkdir(exist_ok=True)
            marker.write_text(now.isoformat())
        except OSError:
            pass
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
