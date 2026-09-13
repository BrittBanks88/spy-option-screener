"""Tests for the read-only price-target view."""
import json

import pytest

pytest.importorskip("pyarrow")


def _view(**kw):
    from spy_option_screener.screener import price_target as pt
    try:
        from spy_option_screener.data import intraday as it
        it.load_intraday("1h")
    except Exception:
        pytest.skip("no cached intraday data")
    return pt.build_view(**kw)


def test_no_lean_path_is_slack_safe():
    v = _view()
    assert isinstance(v.slack_text(), str)
    json.dumps(v.slack_payload())


def test_backcheck_uses_that_days_close_not_live_quote():
    """Regression: spot must come from the historical close for --for, never
    today's live quote."""
    import pandas as pd
    from spy_option_screener.data import loader
    v = _view(for_date="2026-09-03")
    daily = loader.load_history(start="2018-01-01")
    daily.index = pd.DatetimeIndex(daily.index)
    hist_close = float(daily.loc["2026-09-03", "close"])
    assert v.spot == pytest.approx(hist_close)
    assert v.spot_estimated is False


def test_qualified_view_has_ordered_targets_and_no_dilution_bug():
    """Find a session where a single daily signal fires and confirm the
    blended confidence equals that signal's own confidence (not divided by
    the full 5-signal universe -- the dsig.combine() dilution bug)."""
    import pandas as pd
    from spy_option_screener.data import loader, intraday as it
    from spy_option_screener.signals import strategies as st
    daily = loader.load_history(start="2018-01-01")
    daily.index = pd.DatetimeIndex(daily.index)
    bars = it.load_intraday("1h")
    pb = st.pullback(daily)
    pb.index = pd.DatetimeIndex(pb.index)
    cand = pb[(pb["direction"] != 0)
              & (pb.index >= pd.Timestamp(bars["date"].min()))]
    if cand.empty:
        pytest.skip("no pullback signals in the intraday window")
    ts = cand.index[-1]
    v = _view(for_date=ts.date())
    if not v.qualified:
        pytest.skip("didn't qualify at the 30% bar on this date")
    assert v.confidence == pytest.approx(float(cand.loc[ts, "confidence"]), abs=0.02)
    if v.direction > 0:
        assert v.watch_level < v.spot < v.near_target <= v.far_target
    else:
        assert v.far_target <= v.near_target < v.spot < v.watch_level
    json.dumps(v.slack_payload())
    assert len(v.slack_blocks()) <= 5


def test_no_contract_specific_fields_leak_into_output():
    """This is meant to be read-only -- no strike/entry/stop dollar amounts,
    just levels and a lean."""
    v = _view(for_date="2026-09-11")
    text = v.slack_text()
    for banned in ("STOP", "strike", "premium", "limit"):
        assert banned.lower() not in text.lower()
