"""CLI: read-only SPY price-target view -- no contract, no entry, no stop.

    python price_alert.py                     # latest session, mrkdwn to stdout
    python price_alert.py --json              # Slack Block Kit payload
    python price_alert.py --for 2026-09-11     # back-check a past session
    python price_alert.py --webhook <URL>     # POST to an incoming webhook

Informational only. Not a trade recommendation.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

from spy_option_screener import _env
from spy_option_screener.screener.price_target import build_view

_env.load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h", choices=["1h", "30m"])
    ap.add_argument("--for", dest="for_date", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--webhook", default=os.environ.get("SLACK_WEBHOOK_URL"))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    view = build_view(interval=args.interval, for_date=args.for_date,
                      refresh=args.refresh)

    if args.json:
        print(json.dumps(view.slack_payload(), indent=2, ensure_ascii=False))
    else:
        print(view.slack_text())

    if args.webhook:
        body = json.dumps(view.slack_payload()).encode()
        req = urllib.request.Request(args.webhook, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:      # noqa: S310
            print(f"\nposted to Slack — HTTP {resp.status}")


if __name__ == "__main__":
    main()
