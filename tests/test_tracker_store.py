"""Offline tests for the tracker's resolution + exit logic (fake worksheet)."""
import re

import pandas as pd
import pytest

pytest.importorskip("gspread")

from spy_option_screener.data import tracker_store as ts


class FakeWS:
    """Just enough of gspread's Worksheet for tracker_store: A1-style single-row
    updates, get_all_records, append_row, row_values."""

    def __init__(self, rows):
        self.header = list(ts.COLUMNS)
        self.rows = [dict(r) for r in rows]

    def get_all_records(self):
        return [dict(r) for r in self.rows]

    def row_values(self, n):
        return list(self.header)

    def append_row(self, vals):
        self.rows.append(dict(zip(self.header, vals)))

    def update(self, rng, values):
        m = re.fullmatch(r"([A-Z])(\d+)(?::([A-Z])(\d+))?", rng)
        c1, r1 = m.group(1), int(m.group(2))
        c2 = m.group(3) or c1
        row = self.rows[r1 - 2]
        for col, v in zip([chr(x) for x in range(ord(c1), ord(c2) + 1)], values[0]):
            row[self.header[ord(col) - 65]] = v


def _row(i, gap_type, target, stop, time_stop="2026-10-16", status="watching",
         added="2026-09-30"):
    r = {c: "" for c in ts.COLUMNS}
    r.update(id=i, date_added=added, contract=f"test {i}", entry_price=5.0,
             gap_type=gap_type, target=target, stop=stop,
             time_stop_date=time_stop, status=status)
    return r


def _daily(bars):
    idx = pd.to_datetime([b[0] for b in bars])
    return pd.DataFrame({"open": [b[1] for b in bars], "high": [b[2] for b in bars],
                         "low": [b[3] for b in bars], "close": [b[4] for b in bars]},
                        index=idx)


def test_gap_down_hits_target_when_high_reaches_it():
    ws = FakeWS([_row(1, "down", target=100, stop=90)])
    daily = _daily([("2026-09-30", 95, 99, 94, 97), ("2026-10-01", 97, 101, 96, 100)])
    assert ts.refresh_statuses(ws, daily) == 1
    assert ws.rows[0]["status"] == "hit_target"
    assert ws.rows[0]["resolution_date"] == "2026-10-01"


def test_gap_up_hits_stop_when_high_runs_away():
    # gap up: target is BELOW (price must fall), stop is ABOVE
    ws = FakeWS([_row(1, "up", target=100, stop=110)])
    daily = _daily([("2026-09-30", 105, 108, 103, 106), ("2026-10-01", 106, 111, 105, 110)])
    ts.refresh_statuses(ws, daily)
    assert ws.rows[0]["status"] == "hit_stop"


def test_gap_up_hits_target_when_low_falls_to_it():
    ws = FakeWS([_row(1, "up", target=100, stop=110)])
    daily = _daily([("2026-09-30", 105, 106, 99.5, 101)])
    ts.refresh_statuses(ws, daily)
    assert ws.rows[0]["status"] == "hit_target"


def test_times_out_only_after_time_stop_date_passes():
    pending = FakeWS([_row(1, "down", target=100, stop=90, time_stop="2026-10-16")])
    early = _daily([("2026-09-30", 95, 96, 94, 95), ("2026-10-01", 95, 96, 94, 95)])
    assert ts.refresh_statuses(pending, early) == 0
    assert pending.rows[0]["status"] == "watching"

    late = _daily([("2026-09-30", 95, 96, 94, 95), ("2026-10-19", 95, 96, 94, 95)])
    assert ts.refresh_statuses(pending, late) == 1
    assert pending.rows[0]["status"] == "timed_out"


def test_resolved_rows_are_left_alone():
    ws = FakeWS([_row(1, "down", target=100, stop=90, status="hit_stop")])
    daily = _daily([("2026-09-30", 95, 120, 94, 119)])
    assert ts.refresh_statuses(ws, daily) == 0
    assert ws.rows[0]["status"] == "hit_stop"


def test_update_exit_computes_pnl_and_keeps_signal_resolution_separate():
    ws = FakeWS([_row(1, "down", target=100, stop=90)])
    ts.update_exit(ws, "1", exit_price=7.19, exit_date="2026-10-02", entry_price=5.08)
    r = ws.rows[0]
    assert r["entry_price"] == 5.08
    assert r["pnl_dollars"] == pytest.approx(211.0)
    assert r["pnl_pct"] == pytest.approx(7.19 / 5.08 - 1, abs=1e-5)
    assert r["exit_date"] == "2026-10-02"
    # selling early must NOT stamp the signal's own resolution date
    assert r["resolution_date"] == ""
    assert r["status"] == "watching"


def test_trade_result_is_based_on_real_pnl_not_signal_status():
    assert ts.trade_result(7.19, 211.0) == "win"
    assert ts.trade_result(3.00, -208.0) == "loss"
    assert ts.trade_result(5.08, 0) == "flat"
    # not sold yet (blank cells as returned by the sheet) -> no result
    assert ts.trade_result("", "") is None
    assert ts.trade_result(None, None) is None


def test_same_bar_touch_resolves_as_stop_like_the_backtest():
    # both levels inside one bar that did NOT open through the target
    ws = FakeWS([_row(1, "down", target=100, stop=90)])
    daily = _daily([("2026-09-30", 95, 101, 89, 97)])
    ts.refresh_statuses(ws, daily)
    assert ws.rows[0]["status"] == "hit_stop"


def test_accuracy_counts_each_gap_signal_once_not_each_contract():
    rows = [_row(1, "down", 100, 90, status="hit_target"),
            _row(2, "down", 100, 90, status="hit_target"),   # same gap, 2nd contract
            _row(3, "up", 100, 110, status="hit_stop"),
            _row(4, "up", 100, 110, status="watching")]       # still pending
    rows[0]["gap_date"] = rows[1]["gap_date"] = "2026-09-28"
    rows[2]["gap_date"] = rows[3]["gap_date"] = "2026-10-02"
    df = pd.DataFrame(rows)
    assert ts.signal_accuracy(df) == (1, 2)   # 1 hit of 2 distinct resolved signals
