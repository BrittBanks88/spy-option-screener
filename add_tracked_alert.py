"""Add a tracked contract to the gap-screener accuracy log (Google Sheet).

Pulls the currently-open gap signal (target/stop/time-stop) and pairs it with
a real contract + entry price -- typically transcribed from a brokerage
screenshot. Target/stop/time-out get checked automatically later against
real SPY price data (see the app's Tracker tab, or research/... no, that's
data/tracker_store.py:refresh_statuses).

    python add_tracked_alert.py --contract "SPY $765 CALL exp 2026-10-03" \\
        --entry-price 4.25 --date-added 2026-09-30

    # watchlist only -- no money committed, entry-price is just a reference
    # quote (e.g. the mark price at the time), not a fill
    python add_tracked_alert.py --contract "SPY $771 CALL exp 2026-10-16" \\
        --entry-price 5.21 --watching

    python add_tracked_alert.py --exit ID --exit-price 6.10 --exit-date 2026-10-05
    # --entry-price here corrects the cost basis to the broker's own figure
    # before P&L is computed; --notes replaces the row's notes
"""
from __future__ import annotations

import argparse
import datetime as dt

from spy_option_screener.data import tracker_store as ts
from spy_option_screener.screener import gap_screener as gap_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", help='e.g. "SPY $765 CALL exp 2026-10-03"')
    ap.add_argument("--entry-price", type=float)
    ap.add_argument("--date-added", default=dt.date.today().isoformat())
    ap.add_argument("--notes", default="")
    ap.add_argument("--watching", action="store_true",
                    help="watchlist only, not a real position -- entry-price "
                         "is a reference quote, not a fill")
    ap.add_argument("--for", dest="for_date", default=None,
                    help="use the gap signal as of this date instead of today")
    ap.add_argument("--exit", dest="exit_id", help="row id to close out")
    ap.add_argument("--exit-price", type=float)
    ap.add_argument("--exit-date", default=dt.date.today().isoformat())
    args = ap.parse_args()

    client, url = ts.client_from_local_toml()
    ws = ts.get_worksheet(client, url)

    if args.exit_id:
        if args.exit_price is None:
            ap.error("--exit requires --exit-price")
        ts.update_exit(ws, args.exit_id, args.exit_price, args.exit_date,
                       entry_price=args.entry_price,
                       notes=args.notes or None)
        print(f"closed id {args.exit_id} at ${args.exit_price:.2f} on {args.exit_date}")
        return

    if not (args.contract and args.entry_price):
        ap.error("--contract and --entry-price are required to add an entry")

    view = gap_mod.build_view(for_date=args.for_date)
    if not view.qualified:
        print("No open gap signal to attach this contract to "
              f"({view.setup}). Nothing added.")
        return

    status = "watching" if args.watching else "open"
    new_id = ts.add_entry(
        ws, contract=args.contract, entry_price=args.entry_price,
        date_added=args.date_added, gap_date=view.gap_date,
        gap_type=view.gap_type, target=view.target, stop=view.stop,
        time_stop_date=view.time_stop_date, notes=args.notes, status=status,
    )
    print(f"added id {new_id} [{status}]: {args.contract} @ ${args.entry_price:.2f}  ·  "
          f"target ${view.target:.2f}  ·  stop ${view.stop:.2f}  ·  "
          f"time stop {view.time_stop_date}")


if __name__ == "__main__":
    main()
