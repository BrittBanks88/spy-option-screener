"""Entry and exit rule configuration for the long-option backtest."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TradeRules:
    # --- entry ---
    target_dte: int = 5              # buy the nearest weekly with >= this many days
    max_dte: int = 9
    target_delta: float = 0.45       # pick the strike closest to this |delta|
    min_confidence: float = 0.15     # skip weak signals
    entry_time: str | None = None    # e.g. "10:00"; None = enter at the close
    risk_per_trade: float = 0.03     # fraction of equity spent on premium
    max_concurrent: int = 1          # one position at a time (weekly cadence)
    slippage_pct: float = 0.05       # entry+exit haircut vs model mid (each side)
    commission_per_contract: float = 0.65

    # --- exit ---
    profit_target_pct: float = 1.00  # +100% on premium -> take it
    stop_loss_pct: float = 0.60      # -60% on premium -> cut it
    time_stop_dte: int = 1           # close when this many days remain
    exit_on_signal_flip: bool = True # opposite signal -> close

    # --- vol assumptions for reconstruction ---
    vix_baseline_window: int = 40    # days for the regime-tilt baseline
    r: float = 0.04
    q: float = 0.013
