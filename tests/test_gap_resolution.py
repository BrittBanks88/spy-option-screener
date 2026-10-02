"""Shared target/stop resolution rule + the loader's unfinished-bar guard."""
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

from spy_option_screener.data import loader
from spy_option_screener.signals.gaps import first_resolution

ET = ZoneInfo("America/New_York")


def _bars(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({"open": [r[1] for r in rows], "high": [r[2] for r in rows],
                         "low": [r[3] for r in rows], "close": [r[4] for r in rows]},
                        index=idx)


def test_first_level_touched_wins_across_bars():
    # gap DOWN (-1): target 100 above, stop 90 below. Stop is touched first.
    path = _bars([("2026-10-01", 95, 97, 89, 91), ("2026-10-02", 91, 105, 90, 104)])
    outcome, ts = first_resolution(path, -1, target=100, stop=90)
    assert outcome == "stop" and ts == pd.Timestamp("2026-10-01")


def test_same_bar_touch_is_a_stop_unless_the_bar_opened_through_the_target():
    both = _bars([("2026-10-01", 95, 101, 89, 97)])
    assert first_resolution(both, -1, 100, 90)[0] == "stop"      # conservative
    gapped = _bars([("2026-10-01", 101, 102, 89, 97)])
    assert first_resolution(gapped, -1, 100, 90)[0] == "target"  # opened over it


def test_gap_up_mirror():
    # gap UP (+1): target 100 below, stop 110 above
    assert first_resolution(_bars([("2026-10-01", 105, 106, 99, 101)]), 1, 100, 110)[0] == "target"
    assert first_resolution(_bars([("2026-10-01", 105, 111, 103, 110)]), 1, 100, 110)[0] == "stop"
    assert first_resolution(_bars([("2026-10-01", 105, 106, 103, 104)]), 1, 100, 110) == (None, None)


def test_unfinished_todays_bar_is_dropped_until_the_session_is_final():
    day = pd.to_datetime(["2026-10-01", "2026-10-02"])
    df = pd.DataFrame({"Close": [1.0, 2.0]}, index=day)
    midday = dt.datetime(2026, 10, 2, 12, 33, tzinfo=ET)
    after = dt.datetime(2026, 10, 2, 16, 20, tzinfo=ET)
    assert list(loader._drop_unfinished_today(df, now=midday).index.date) == [dt.date(2026, 10, 1)]
    assert len(loader._drop_unfinished_today(df, now=after)) == 2
