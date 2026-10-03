"""Contract picker, run against the FAKE chain (no data subscription needed)."""
import datetime as dt

import pytest

from spy_option_screener.data.fake_chain import REQUIRED_COLUMNS, fake_chain
from spy_option_screener.screener.contract_picker import (PickerRules,
                                                          suggest_contract)

SPOT = 769.40
TODAY = dt.date(2026, 10, 5)
UP = dict(gap_type="up", spot=SPOT, target=763.99, stop=783.83)       # expect a fall -> put
DOWN = dict(gap_type="down", spot=SPOT, target=775.0, stop=755.08)    # expect a rise -> call


@pytest.fixture(scope="module")
def chain():
    return fake_chain(SPOT, today=TODAY)


def test_fake_chain_follows_the_documented_schema(chain):
    assert set(REQUIRED_COLUMNS) <= set(chain.columns)
    assert chain.attrs["source"] == "fake" and chain.attrs["spot"] == SPOT
    assert (chain["ask"] > chain["bid"]).all()


def test_gap_up_picks_a_put_and_gap_down_picks_a_call(chain):
    up = suggest_contract(chain, budget=(500, 1500), hold_days=14, **UP).suggestion
    down = suggest_contract(chain, budget=(500, 1500), hold_days=14, **DOWN).suggestion
    assert up.kind == "put" and down.kind == "call"


def test_suggestion_respects_budget_and_expiry_window(chain):
    rules = PickerRules()
    for hold in (7, 14):
        s = suggest_contract(chain, budget=(500, 1500), hold_days=hold, **UP).suggestion
        assert 500 <= s.cost <= 1500
        assert s.cost == pytest.approx(s.ask * 100)           # you pay the ASK
        assert hold + rules.expiry_cushion_days <= s.dte <= (
            hold + rules.expiry_cushion_days + rules.max_extra_dte)


def test_later_hits_pay_less_because_of_time_decay(chain):
    s = suggest_contract(chain, budget=(500, 1500), hold_days=14, **UP).suggestion
    profits = [x.profit_at_target for x in s.scenarios]
    assert profits == sorted(profits, reverse=True)
    assert [x.days_to_hit for x in s.scenarios] == [1, 5, 10]


def test_target_is_profitable_early_and_stop_loses_but_never_more_than_the_premium(chain):
    s = suggest_contract(chain, budget=(500, 1500), hold_days=14, **UP).suggestion
    assert s.scenarios[0].profit_at_target > 0
    for x in s.scenarios:
        assert -s.cost - 1e-6 <= x.loss_at_stop < 0


def test_no_contract_fits_returns_a_reason_not_a_forced_pick(chain):
    r = suggest_contract(chain, budget=(5, 20), hold_days=14, **UP)
    assert r.suggestion is None
    assert "Nothing priced between $5 and $20" in r.reason
    assert "eligible contracts run" in r.reason


def test_illiquid_chain_is_rejected(chain):
    thin = chain.copy()
    thin["open_interest"] = 0
    r = suggest_contract(thin, budget=(500, 1500), hold_days=14, **UP)
    assert r.suggestion is None and "liquidity" in r.reason


def test_expiry_window_failure_is_explained(chain):
    r = suggest_contract(chain, budget=(500, 1500), hold_days=90, **UP)
    assert r.suggestion is None and "No expiry between" in r.reason


def test_warnings_flag_fake_data_and_modeled_pnl(chain):
    s = suggest_contract(chain, budget=(500, 1500), hold_days=14, **UP).suggestion
    assert any("FAKE CHAIN" in w for w in s.warnings)
    assert any("model estimate" in w for w in s.warnings)


def test_stale_quotes_are_flagged(chain):
    stale = chain.copy()
    stale.attrs = {**chain.attrs, "quote_age_seconds": 600.0}
    s = suggest_contract(stale, budget=(500, 1500), hold_days=14, **UP).suggestion
    assert any("600 seconds old" in w for w in s.warnings)


def test_missing_columns_and_bad_gap_type_fail_loudly(chain):
    with pytest.raises(ValueError, match="missing required columns"):
        suggest_contract(chain.drop(columns=["iv"]), budget=(500, 1500),
                         hold_days=14, **UP)
    with pytest.raises(ValueError, match="gap_type"):
        suggest_contract(chain, gap_type="sideways", spot=SPOT, target=1, stop=2,
                         budget=(500, 1500), hold_days=14)


def test_iv_crush_lowers_the_modeled_profit(chain):
    base = suggest_contract(chain, budget=(500, 1500), hold_days=14, **UP).suggestion
    crushed = suggest_contract(chain, budget=(500, 1500), hold_days=14,
                               rules=PickerRules(iv_shift=-0.03), **UP).suggestion
    assert crushed.headline.profit_at_target < base.headline.profit_at_target
