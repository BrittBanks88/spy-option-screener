"""CLI: SPY gap-fill screener -- price target, stop, and time-stop, no contract.

    python gap_alert.py                     # latest session, mrkdwn to stdout
    python gap_alert.py --json              # Slack Block Kit payload
    python gap_alert.py --for 2026-09-28    # back-check a past session
    python gap_alert.py --webhook <URL>     # POST to an incoming webhook

Backtested on 26+ years of SPY daily bars (research/backtest_gap_screener.py):
85% hit target, 14% stop out, 1% time out within 14 trading days.
Informational only. Not a trade recommendation.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

from spy_option_screener import _env
from spy_option_screener.screener.gap_screener import build_view

_env.load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--for", dest="for_date", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--webhook", default=os.environ.get("SLACK_WEBHOOK_URL"))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    view = build_view(for_date=args.for_date, refresh=args.refresh)

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
