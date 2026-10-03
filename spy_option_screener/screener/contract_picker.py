"""Turn a live gap alert + a budget into ONE suggested option contract, with
the modeled profit if SPY reaches the target and loss if it hits the stop.

SKELETON: the logic is real and tested, but it has only ever run on a FAKE
chain (data/fake_chain.py). The Signal-tab card already calls it, on that fake
chain, and says so. The rules below are sensible starting points -- not
validated ones. See HANDOFF_CONTRACT_PICKER.md for what's left.

Direction: a gap UP is expected to fall back to the pre-gap level -> buy a PUT;
a gap DOWN is expected to rise back -> buy a CALL.

How profit/loss is estimated (and why it is only an estimate):
  * Cost is the ASK x 100 -- what you would actually pay.
  * The option is re-priced with Black-Scholes at the target (or stop) price,
    N days later, using the contract's own IV. Rather than trust the model's
    absolute level, we take the model's CHANGE in value and add it to today's
    market mid, so a small model-vs-market gap doesn't bias the result.
  * Selling is assumed to give up half the quoted spread.
  * Several timing scenarios are returned because WHEN SPY gets there matters:
    the same move pays less on day 10 than on day 1 (time decay).
  * IV is held constant unless ``rules.iv_shift`` says otherwise. The 85%
    hit-rate backtest is about SPY's PRICE; the option P&L has NOT been
    backtested, so treat every dollar figure as a model estimate.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.fake_chain import REQUIRED_COLUMNS
from ..pricing import black_scholes as bs


@dataclass(frozen=True)
class PickerRules:
    """Every threshold in one place. All are starting points to be tuned."""
    delta_band: tuple[float, float] = (0.35, 0.60)   # |delta|; same band the earlier plan used
    min_open_interest: int = 100
    max_spread_pct: float = 0.08                     # (ask-bid)/mid
    expiry_cushion_days: int = 3                     # expiry must outlast the hold by this much
    max_extra_dte: int = 14                          # ...but not by more than this
    scenario_days: tuple[int, int, int] = (1, 5, 10)  # days until SPY reaches target/stop
    iv_shift: float = 0.0                            # added to IV when re-pricing (e.g. -0.02 = IV drops 2 pts)
    r: float = 0.04
    q: float = 0.013


@dataclass
class Scenario:
    days_to_hit: int
    profit_at_target: float      # dollars per contract, after spread; can be negative
    loss_at_stop: float          # dollars per contract; never worse than -cost


@dataclass
class ContractSuggestion:
    kind: str                    # "call" | "put"
    strike: float
    expiry: dt.date
    dte: int
    bid: float
    ask: float
    cost: float                  # dollars per contract (ask x 100)
    iv: float
    delta: float
    open_interest: int | None
    spread_pct: float
    scenarios: list[Scenario]
    headline: Scenario           # the middle scenario -- what the page should lead with
    warnings: list[str] = field(default_factory=list)

    @property
    def profit_pct(self) -> float:
        return self.headline.profit_at_target / self.cost

    @property
    def loss_pct(self) -> float:
        return self.headline.loss_at_stop / self.cost


@dataclass
class PickResult:
    suggestion: ContractSuggestion | None
    reason: str = ""             # why there is no suggestion (empty when there is one)
    n_candidates: int = 0        # contracts that passed every filter


def _model(kind: str, spot: float, strike: float, dte: float, iv: float,
           r: float, q: float) -> float:
    t = max(dte, 0.5) / 365.0
    return float(bs.price(spot, strike, t, iv, r, q, kind))


def _scenarios(row, kind: str, spot: float, target: float, stop: float,
               rules: PickerRules) -> list[Scenario]:
    iv = float(row["iv"]) + rules.iv_shift
    iv0 = float(row["iv"])
    dte = float(row["dte"])
    strike = float(row["strike"])
    mid_now = float(row["mid"])
    ask = float(row["ask"])
    half_spread = float(row["spread_pct"]) / 2.0
    v0 = _model(kind, spot, strike, dte, iv0, rules.r, rules.q)
    out = []
    for d in rules.scenario_days:
        rem = dte - d
        exits = []
        for level in (target, stop):
            v1 = _model(kind, level, strike, rem, iv, rules.r, rules.q)
            exit_mid = max(mid_now + (v1 - v0), 0.0)
            exits.append(exit_mid * (1.0 - half_spread))
        profit = (exits[0] - ask) * 100.0
        loss = max((exits[1] - ask) * 100.0, -ask * 100.0)
        out.append(Scenario(int(d), float(profit), float(loss)))
    return out


def suggest_contract(chain: pd.DataFrame, *, gap_type: str, spot: float,
                     target: float, stop: float, budget: tuple[float, float],
                     hold_days: int, rules: PickerRules = PickerRules()) -> PickResult:
    """Pick the best contract in ``chain`` for this alert and budget.

    chain     : standard schema, see data/fake_chain.py
    gap_type  : "up" -> put, "down" -> call
    budget    : (low, high) dollars to spend on ONE contract (ask x 100)
    hold_days : how long the user intends to hold; the expiry must outlast it
    Returns a PickResult; ``suggestion`` is None when nothing qualifies, with
    ``reason`` saying which filter emptied the list (so the page can tell the
    user to widen the budget instead of showing a blank).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in chain.columns]
    if missing:
        raise ValueError(f"chain is missing required columns: {missing}")
    if gap_type not in ("up", "down"):
        raise ValueError("gap_type must be 'up' or 'down'")
    kind = "put" if gap_type == "up" else "call"
    lo, hi = budget

    c = chain[chain["type"] == kind].copy()
    if c.empty:
        return PickResult(None, f"The chain has no {kind}s.")

    min_dte = hold_days + rules.expiry_cushion_days
    c = c[(c["dte"] >= min_dte) & (c["dte"] <= min_dte + rules.max_extra_dte)]
    if c.empty:
        return PickResult(None, f"No expiry between {min_dte} and "
                                f"{min_dte + rules.max_extra_dte} days out "
                                f"(needed to outlast a {hold_days}-day hold).")

    if "mid" not in c.columns:
        c["mid"] = (c["bid"] + c["ask"]) / 2.0
    c = c[(c["bid"] > 0) & (c["ask"] > 0) & (c["iv"] > 0.01)].copy()
    if c.empty:
        return PickResult(None, "No contracts with a usable bid, ask and IV.")

    if "delta" not in c.columns:
        t = np.maximum(c["dte"].values, 0.5) / 365.0
        c["delta"] = bs.greeks(spot, c["strike"].values, t, c["iv"].values,
                               rules.r, rules.q, kind)["delta"]
    c = c[c["delta"].abs().between(*rules.delta_band)]
    if c.empty:
        return PickResult(None, f"No {kind}s with |delta| between "
                                f"{rules.delta_band[0]} and {rules.delta_band[1]}.")

    c["spread_pct"] = (c["ask"] - c["bid"]) / c["mid"]
    has_oi = "open_interest" in c.columns
    c = c[c["spread_pct"] <= rules.max_spread_pct]
    if has_oi:
        c = c[c["open_interest"] >= rules.min_open_interest]
    if c.empty:
        return PickResult(None, "Every candidate failed the liquidity checks "
                                f"(spread over {rules.max_spread_pct:.0%} or "
                                f"open interest under {rules.min_open_interest}).")

    c["cost"] = c["ask"] * 100.0
    in_budget = c[(c["cost"] >= lo) & (c["cost"] <= hi)]
    if in_budget.empty:
        return PickResult(None, f"Nothing priced between ${lo:,.0f} and "
                                f"${hi:,.0f}; eligible contracts run "
                                f"${c['cost'].min():,.0f} to ${c['cost'].max():,.0f}.")

    best, best_key = None, None
    for _, row in in_budget.iterrows():
        scen = _scenarios(row, kind, spot, target, stop, rules)
        headline = scen[len(scen) // 2]
        # Objective (a DECISION POINT, see the handoff doc): best modeled % gain
        # at the target, ties broken by tighter spread then deeper open interest.
        key = (headline.profit_at_target / row["cost"], -row["spread_pct"],
               float(row["open_interest"]) if has_oi else 0.0)
        if best_key is None or key > best_key:
            best_key, best = key, (row, scen, headline)

    row, scen, headline = best
    warnings = ["Profit and loss are a model estimate (re-priced with the "
                "contract's own IV, spread included). The option P&L for these "
                "alerts has not been backtested."]
    if chain.attrs.get("source") == "fake":
        warnings.insert(0, "FAKE CHAIN: development data, not market quotes.")
    age = chain.attrs.get("quote_age_seconds")
    if age is not None and age > 90:
        warnings.append(f"Quotes are {age:.0f} seconds old.")
    if not has_oi:
        warnings.append("No open-interest data, so liquidity was only checked by spread.")
    if row["spread_pct"] > 0.05:
        warnings.append(f"Wide bid-ask spread ({row['spread_pct']:.1%}); some of "
                        "the move is lost to it on the way in and out.")
    if headline.profit_at_target <= 0:
        warnings.append("At this timing, even reaching the target does not "
                        "pay after time decay and spread.")

    return PickResult(ContractSuggestion(
        kind=kind, strike=float(row["strike"]),
        expiry=pd.Timestamp(row["expiry"]).date() if "expiry" in row else None,
        dte=int(row["dte"]), bid=float(row["bid"]), ask=float(row["ask"]),
        cost=float(row["cost"]), iv=float(row["iv"]), delta=float(row["delta"]),
        open_interest=int(row["open_interest"]) if has_oi else None,
        spread_pct=float(row["spread_pct"]), scenarios=scen, headline=headline,
        warnings=warnings), n_candidates=len(in_budget))
