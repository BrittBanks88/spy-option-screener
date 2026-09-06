"""Event-driven daily backtest for long SPY weekly options.

Loop, per trading day t (using close-of-day data, so signals from t-1 are
actionable without look-ahead):

  1. If a position is open, re-price it from today's spot + VIX + remaining
     days, then check exits (profit target, stop, time-stop, signal flip,
     expiry). Realize P&L on exit.
  2. If flat and yesterday's signal is actionable, open one position: build a
     synthetic chain, pick the strike nearest target delta, size by premium
     risk, pay slippage + commission.

Option prices throughout come from Black-Scholes on a VIX-derived vol surface
(pricing.vol_model) -- approximate, see that module's docstring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..pricing import black_scholes as bs
from ..pricing import vol_model as vm
from ..screener import chain as chain_mod
from .rules import TradeRules


@dataclass
class OpenPosition:
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    kind: str
    strike: float
    contracts: int
    entry_price: float          # per share, after slippage
    entry_spot: float
    direction: int
    confidence: float
    note: str
    cost: float                 # total debit incl commission


def _pick_expiry(index, i, rules: TradeRules):
    """Return (expiry_timestamp, calendar_dte) for a weekly ~target_dte out."""
    j = i
    while j < len(index) - 1:
        j += 1
        cal = (index[j] - index[i]).days
        if cal >= rules.target_dte:
            if cal > rules.max_dte:
                return None, None
            return index[j], cal
    return None, None


def _price_option(spot, vix, vix_base, dte, strike, kind, rules: TradeRules):
    t = bs.years_from_days(dte, basis="calendar")
    if dte <= 0:
        return max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    iv = float(vm.skew_adjust(
        vm.atm_vol_from_vix(vix, vix_base), strike, spot, dte))
    return float(bs.price(spot, strike, t, iv, rules.r, rules.q, kind))


def run(df: pd.DataFrame, signals: pd.DataFrame, rules: TradeRules | None = None,
        starting_capital: float = 25_000.0, warmup: int = 210,
        entry_prices: pd.Series | None = None, same_day_signal: bool = False):
    """Return (equity: Series, trades: DataFrame, marks: DataFrame).

    entry_prices : optional {date -> SPY price} to open at instead of the close
                   (e.g. the 10:00 intraday price). Missing/NaN dates fall back
                   to the close.
    same_day_signal : if True the signal is known intraday and acted on the
                      SAME day (no 1-day shift). Use with intraday signals.
    """
    rules = rules or TradeRules()
    df = df.copy()
    df["vix_base"] = df["vix"].rolling(rules.vix_baseline_window).mean()
    sig = signals.reindex(df.index).fillna({"direction": 0, "confidence": 0.0})
    shift = 0 if same_day_signal else 1
    sig_dir = sig["direction"].shift(shift).fillna(0).astype(int)
    sig_conf = sig["confidence"].shift(shift).fillna(0.0)
    sig_note = sig["edge_note"].shift(shift).fillna("")

    if entry_prices is not None:
        entry_prices = entry_prices.copy()
        entry_prices.index = pd.DatetimeIndex(entry_prices.index)

    index = df.index
    cash = starting_capital
    pos: OpenPosition | None = None
    equity_curve = []
    trades = []
    marks = []

    for i in range(len(index)):
        date = index[i]
        row = df.iloc[i]
        spot, vix, vix_base = row["close"], row["vix"], row["vix_base"]
        if np.isnan(vix_base):
            equity_curve.append((date, cash))
            continue

        # ---- 1. manage open position --------------------------------------
        pos_value = 0.0
        if pos is not None:
            dte_left = (pos.expiry_date - date).days
            opt_px = _price_option(spot, vix, vix_base, dte_left,
                                   pos.strike, pos.kind, rules)
            pos_value = opt_px * 100 * pos.contracts
            pnl_pct = opt_px / pos.entry_price - 1.0

            reason = None
            if date >= pos.expiry_date or dte_left <= 0:
                reason = "expiry"
            elif pnl_pct >= rules.profit_target_pct:
                reason = "profit_target"
            elif pnl_pct <= -rules.stop_loss_pct:
                reason = "stop_loss"
            elif dte_left <= rules.time_stop_dte:
                reason = "time_stop"
            elif (rules.exit_on_signal_flip and sig_dir.iloc[i] != 0
                  and sig_dir.iloc[i] != pos.direction):
                reason = "signal_flip"

            if reason:
                exit_px = opt_px * (1.0 - rules.slippage_pct) if reason != "expiry" else opt_px
                proceeds = exit_px * 100 * pos.contracts
                # No closing commission on an option left to expire worthless.
                worthless_expiry = reason == "expiry" and exit_px <= 0.005
                commission = 0.0 if worthless_expiry else rules.commission_per_contract * pos.contracts
                cash += proceeds - commission
                pnl = proceeds - commission - pos.cost
                trades.append({
                    "entry_date": pos.entry_date,
                    "exit_date": date,
                    "type": pos.kind,
                    "strike": pos.strike,
                    "contracts": pos.contracts,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_px,
                    "entry_spot": pos.entry_spot,
                    "exit_spot": spot,
                    "hold_days": (date - pos.entry_date).days,
                    "pnl": pnl,
                    "return_pct": pnl / pos.cost,
                    "exit_reason": reason,
                    "confidence": pos.confidence,
                    "note": pos.note,
                })
                pos, pos_value = None, 0.0

        # ---- 2. open a new position --------------------------------------
        if pos is None and sig_dir.iloc[i] != 0 and sig_conf.iloc[i] >= rules.min_confidence:
            direction = int(sig_dir.iloc[i])
            expiry, dte = _pick_expiry(index, i, rules)
            if expiry is not None:
                kind = "call" if direction > 0 else "put"
                entry_spot = spot
                if entry_prices is not None and date in entry_prices.index:
                    ep = float(entry_prices.loc[date])
                    if np.isfinite(ep) and ep > 0:
                        entry_spot = ep
                ch = chain_mod.synthetic_chain(entry_spot, vix, dte,
                                               vix_baseline=vix_base,
                                               r=rules.r, q=rules.q)
                side = ch[ch["type"] == kind].copy()
                side["ddist"] = (side["delta"].abs() - rules.target_delta).abs()
                pick = side.sort_values("ddist").iloc[0]

                fill = pick["mid"] * (1.0 + rules.slippage_pct)
                budget = cash * rules.risk_per_trade
                contracts = int(budget // (fill * 100))
                if contracts >= 1:
                    commission = rules.commission_per_contract * contracts
                    cost = fill * 100 * contracts + commission
                    if cost <= cash:
                        cash -= cost
                        pos = OpenPosition(
                            entry_date=date, expiry_date=expiry, kind=kind,
                            strike=float(pick["strike"]), contracts=contracts,
                            entry_price=float(fill), entry_spot=float(entry_spot),
                            direction=direction, confidence=float(sig_conf.iloc[i]),
                            note=str(sig_note.iloc[i]), cost=cost,
                        )
                        pos_value = pick["mid"] * 100 * contracts

        equity_curve.append((date, cash + pos_value))
        marks.append({"date": date, "cash": cash, "position_value": pos_value,
                      "in_market": pos is not None})

    equity = pd.Series({d: v for d, v in equity_curve}, name="equity")
    equity.index = pd.DatetimeIndex(equity.index)
    trades_df = pd.DataFrame(trades)
    marks_df = pd.DataFrame(marks).set_index("date") if marks else pd.DataFrame()
    return equity.iloc[warmup:] if len(equity) > warmup else equity, trades_df, marks_df
