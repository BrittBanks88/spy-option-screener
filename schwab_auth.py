"""One-time / weekly Schwab OAuth helper.

    python schwab_auth.py            # full browser authorise -> save token
    python schwab_auth.py --refresh  # just refresh the access token
    python schwab_auth.py --status   # how long the tokens are good for

Needs SCHWAB_APP_KEY / SCHWAB_APP_SECRET / SCHWAB_CALLBACK_URL in .env.
The refresh token lasts 7 days, so re-run the plain command about weekly.
"""
from __future__ import annotations

import argparse
import webbrowser

from spy_option_screener import _env
from spy_option_screener.data import schwab

_env.load()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        s = schwab.token_status()
        if not s["have_token"]:
            print("no token — run: python schwab_auth.py")
            return 1
        print(f"access token valid for  {s['access_valid_for']/60:.0f} min")
        print(f"refresh token valid for {s['refresh_valid_for_days']:.1f} days")
        return 0

    if args.refresh:
        schwab.refresh()
        print("access token refreshed.")
        return 0

    url = schwab.authorize_url()
    print("\n1. Opening Schwab in your browser (or copy this):\n")
    print("   " + url + "\n")
    try:
        webbrowser.open(url)
    except Exception:      # noqa: BLE001
        pass
    print("2. Log in and approve. Your browser will redirect to a URL that starts")
    print("   with your callback (it may show a 'can't reach this page' error —")
    print("   that's fine, the code is in the address bar).\n")
    redirect = input("3. Paste the FULL redirected URL here:\n   ").strip()
    schwab.exchange_code(redirect)
    s = schwab.token_status()
    print(f"\n✓ saved. refresh token good for {s['refresh_valid_for_days']:.1f} days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
