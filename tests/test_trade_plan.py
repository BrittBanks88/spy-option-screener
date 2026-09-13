"""Tests for the pullback→continuation trade plan and its Slack payload."""
import json

import pytest

pytest.importorskip("pyarrow")


def _plan(**kw):
    from spy_option_screener.screener import trade_plan as tp
    try:
        from spy_option_screener.data import intraday as it
        it.load_intraday("1h")
    except Exception:
        pytest.skip("no cached intraday data")
    return tp.build_plan(**kw)


def test_no_setup_path_is_slack_safe():
    p = _plan()                                   # latest session
    assert isinstance(p.slack_text(), str) and len(p.slack_text()) > 20
    payload = p.slack_payload()
    assert set(payload) == {"text", "blocks"}
    json.dumps(payload)                            # must serialise


def test_qualified_plan_is_internally_consistent():
    """Find a session where it qualifies and check the levels line up."""
    import pandas as pd
    from spy_option_screener.data import loader, intraday as it
    from spy_option_screener.signals import strategies as st

    daily = loader.load_history(start="2018-01-01")
    daily.index = pd.DatetimeIndex(daily.index)
    bars = it.load_intraday("1h")
    pb = st.pullback(daily)
    pb.index = pd.DatetimeIndex(pb.index)
    cand = pb[(pb["direction"] != 0)
              & (pb.index >= pd.Timestamp(bars["date"].min()))].index
    if len(cand) == 0:
        pytest.skip("no pullback signals in the intraday window")

    got = None
    for ts in cand[::-1]:
        pl = _plan(for_date=ts.date())
        if pl.qualified:
            got = pl
            break
    if got is None:
        pytest.skip("no qualified plan found")

    if got.direction > 0:                          # call
        assert got.u_stop < got.u_entry < got.u_t1 <= got.u_run
        assert got.o_stop < got.o_entry < got.o_t1 <= got.o_run
    else:                                          # put
        assert got.u_run <= got.u_t1 < got.u_entry < got.u_stop
        assert got.o_stop < got.o_entry < got.o_t1 <= got.o_run
    assert abs(got.o_stop_pct + 0.5) < 1e-9        # stated stop is -50%
    assert got.max_loss == pytest.approx(got.o_entry * 100)
    assert got.reward_risk > 0
    json.dumps(got.slack_payload())
    # blocks are well-formed
    blocks = got.slack_blocks()
    assert blocks[0]["type"] == "header"
    assert all("type" in b for b in blocks)

    # simple version: short, has the 5 action lines, serialises
    simple = got.slack_simple_text()
    assert simple.count("\n") < 14
    for label in ("*GO IF*", "*BUY*", "*STOP*", "*TARGET*", "*DONE BY*"):
        assert label in simple
    sp = got.slack_payload(style="simple")
    json.dumps(sp)
    assert len(sp["blocks"]) <= 3


def test_overnight_assessment_bounds_and_multi_night_penalty():
    import datetime as dt
    import pandas as pd
    from spy_option_screener.data import loader
    from spy_option_screener.screener import overnight as ov
    d = loader.load_history(start="2018-01-01")
    v = ov.assess(d)
    assert v.rating in ("favourable", "neutral", "avoid")
    assert -1.0 <= v.score <= 1.0
    assert v.exp_move_pct > 0 and v.calendar_nights >= 1
    assert "night" in v.line().lower() or "carry" in v.line().lower()
    # a Friday close (weekend carry) scores <= the same setup on a weeknight
    fri = ov.assess(d, dt.date(2026, 8, 21))          # Friday
    wed = ov.assess(d, dt.date(2026, 8, 19))          # Wednesday
    assert fri.calendar_nights >= 3 and wed.calendar_nights == 1


def test_run_alert_dry_run_smoke(capsys):
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "run_alert.py", "--mode", "morning", "--dry-run",
         "--force", "--no-dedupe"],
        capture_output=True, text=True, timeout=180)
    assert r.returncode == 0
    assert "mode=morning" in r.stdout


def _run_alert(*extra_args, timeout=180):
    import subprocess
    import sys
    return subprocess.run(
        [sys.executable, "run_alert.py", "--dry-run", "--no-dedupe", *extra_args],
        capture_output=True, text=True, timeout=timeout)


def test_explicit_mode_ignores_wall_clock_window():
    """A cron-triggered run must not depend on landing inside the narrow
    auto-mode window — GitHub's scheduler can fire hours late. Explicit
    --mode (what the workflow now passes) must run regardless of the clock."""
    r = _run_alert("--mode", "close", "--ignore-window", "--force")
    assert r.returncode == 0
    assert "outside the alert windows" not in r.stdout
    assert "mode=close" in r.stdout


def test_auto_mode_outside_window_without_ignore_flag_is_a_noop():
    r = _run_alert("--mode", "auto")
    # on a non-trading day this exits for that reason instead; either way it
    # must not silently crash, and it must not report post=True
    assert r.returncode == 0
    assert "post=True" not in r.stdout


def test_ignore_window_still_respects_trading_day_gate():
    """--ignore-window is not --force: a holiday/weekend must still no-op."""
    import datetime as dt
    from spy_option_screener.data import market_calendar as mcal
    if mcal.is_trading_day(dt.date.today()):
        pytest.skip("today is a trading day; this checks the holiday gate")
    r = _run_alert("--mode", "morning", "--ignore-window")
    assert r.returncode == 0
    assert "not a trading day" in r.stdout


def test_slack_text_has_the_five_plan_lines():
    import pandas as pd
    from spy_option_screener.data import loader, intraday as it
    from spy_option_screener.signals import strategies as st
    daily = loader.load_history(start="2018-01-01")
    daily.index = pd.DatetimeIndex(daily.index)
    bars = it.load_intraday("1h")
    pb = st.pullback(daily)
    pb.index = pd.DatetimeIndex(pb.index)
    cand = pb[(pb["direction"] != 0)
              & (pb.index >= pd.Timestamp(bars["date"].min()))].index
    for ts in cand[::-1]:
        pl = _plan(for_date=ts.date())
        if pl.qualified:
            txt = pl.slack_text()
            for label in ("*Contract:*", "*Entry:*", "*Stop:*", "*Target 1:*",
                          "*Runner:*", "*Time stop:*", "*Risk:*"):
                assert label in txt
            return
    pytest.skip("no qualified plan")
