"""CLI: the pullback→continuation trade plan, formatted for Slack.

    python slack_plan.py                     # latest session, mrkdwn to stdout
    python slack_plan.py --json              # Slack Block Kit payload (JSON)
    python slack_plan.py --for 2026-09-02    # a past session (demo / check)
    python slack_plan.py --webhook <URL>     # POST to an incoming webhook

The webhook POST sends a message on your behalf — it only runs when you pass
--webhook explicitly with your own URL. Not investment advice.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

from spy_option_screener import _env
from spy_option_screener.screener.trade_plan import build_plan

_env.load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="10:00", help="decision/entry time ET")
    ap.add_argument("--interval", default="1h", choices=["1h", "30m"])
    ap.add_argument("--for", dest="for_date", default=None,
                    help="YYYY-MM-DD of a past session (must have intraday bars)")
    ap.add_argument("--json", action="store_true", help="print the Block Kit payload")
    ap.add_argument("--full", action="store_true",
                    help="detailed card (default is the simple one)")
    ap.add_argument("--webhook", default=os.environ.get("SLACK_WEBHOOK_URL"),
                    help="Slack incoming-webhook URL (defaults to $SLACK_WEBHOOK_URL / .env)")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    plan = build_plan(entry_time=args.entry, interval=args.interval,
                      for_date=args.for_date, refresh=args.refresh)
    style = "full" if args.full else "simple"

    if args.json:
        print(json.dumps(plan.slack_payload(style=style), indent=2, ensure_ascii=False))
    else:
        print(plan.slack_text() if args.full else plan.slack_simple_text())

    if args.webhook:
        body = json.dumps(plan.slack_payload(style=style)).encode()
        req = urllib.request.Request(args.webhook, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:      # noqa: S310
            print(f"\nposted to Slack — HTTP {resp.status}")


if __name__ == "__main__":
    main()
