"""Minimal NYSE trading-day calendar (no external deps).

USFederalHolidayCalendar minus the two federal holidays the NYSE ignores
(Columbus Day, Veterans Day), plus Good Friday. Good enough to pick a valid
option expiration date; not a substitute for pandas_market_calendars.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import pandas as pd
from pandas.tseries.holiday import (USFederalHolidayCalendar, GoodFriday,
                                    Holiday, nearest_workday)


class _NYSE(USFederalHolidayCalendar):
    rules = [r for r in USFederalHolidayCalendar.rules
             if r.name not in ("Columbus Day", "Veterans Day")] + [GoodFriday]


@lru_cache(maxsize=8)
def _holidays(y0: int, y1: int):
    return set(_NYSE().holidays(f"{y0}-01-01", f"{y1}-12-31").date)


def is_trading_day(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in _holidays(d.year - 1, d.year + 1)


def next_trading_day(d: dt.date) -> dt.date:
    d += dt.timedelta(days=1)
    while not is_trading_day(d):
        d += dt.timedelta(days=1)
    return d


def next_weekly_expiry(session_date: dt.date, min_dte: int = 2,
                       max_dte: int = 9) -> tuple[dt.date, int]:
    """First tradable SPY expiration at least ``min_dte`` calendar days out.

    SPY lists expirations every trading day; we just need a real trading day.
    """
    d = session_date
    while True:
        d += dt.timedelta(days=1)
        dte = (d - session_date).days
        if is_trading_day(d) and dte >= min_dte:
            return d, dte
        if dte > max_dte + 4:
            return d, dte
