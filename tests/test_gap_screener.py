"""Tests for the SPY gap-fill screener."""
import json

import pytest

pytest.importorskip("pyarrow")


def _view(**kw):
    from spy_option_screener.screener import gap_screener as gs
    return gs.build_view(**kw)


def test_output_is_slack_safe():
    v = _view()
    assert isinstance(v.slack_text(), str)
    json.dumps(v.slack_payload())


def test_backcheck_uses_that_days_close_not_live_quote():
    """Regression: spot must come from the historical close for --for, never
    today's live quote (same class of bug hit in price_target.py earlier)."""
    import pandas as pd
    from spy_option_screener.data import loader
    from spy_option_screener.screener import gap_screener as gs

    daily = loader.load_history(start="2000-01-01")
    daily.index = pd.DatetimeIndex(daily.index)
    # walk back from the end to find a date with an open (unresolved) gap
    target_date = None
    for i in range(len(daily) - 1, len(daily) - 200, -1):
        v = _view(for_date=str(daily.index[i].date()))
        if v.qualified:
            target_date = daily.index[i].date()
            break
    if target_date is None:
        pytest.skip("no qualifying open gap found in the recent window")

    v = _view(for_date=str(target_date))
    hist_close = float(daily.loc[str(target_date), "close"])
    assert v.spot == pytest.approx(hist_close)
    assert v.spot_estimated is False


def test_qualified_view_has_consistent_target_stop_ordering():
    """By construction: gap up -> target (prior close) sits below the gap
    open, stop sits above it. Gap down -> mirrored. True regardless of
    where price has wandered since, so this is a solid regression check."""
    import pandas as pd
    from spy_option_screener.data import loader
    from spy_option_screener.screener import gap_screener as gs

    daily = loader.load_history(start="2000-01-01")
    daily.index = pd.DatetimeIndex(daily.index)
    found = False
    for i in range(len(daily) - 1, len(daily) - 200, -1):
        v = _view(for_date=str(daily.index[i].date()))
        if not v.qualified:
            continue
        found = True
        if v.gap_type == "up":
            assert v.target < v.gap_open <= v.stop
        else:
            assert v.target > v.gap_open >= v.stop
        assert 0 <= v.days_elapsed <= 14
        assert 0 <= v.historical_fill_rate <= 1
    if not found:
        pytest.skip("no qualifying open gap found in the recent window")


def test_no_contract_specific_fields_leak_into_output():
    """Read-only price/time levels only -- no strike/premium/option mechanics."""
    v = _view()
    text = v.slack_text()
    for banned in ("strike", "premium", "limit", "delta", "contract"):
        assert banned.lower() not in text.lower()
