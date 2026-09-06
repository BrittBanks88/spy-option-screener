"""Performance statistics for an equity curve and a trade log."""
from __future__ import annotations

import numpy as np
import pandas as pd


def equity_stats(equity: pd.Series, freq: int = 252) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / freq
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    ann_vol = rets.std() * np.sqrt(freq)
    sharpe = (rets.mean() * freq) / ann_vol if ann_vol > 0 else np.nan
    downside = rets[rets < 0].std() * np.sqrt(freq)
    sortino = (rets.mean() * freq) / downside if downside > 0 else np.nan

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = dd.min()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": (cagr / abs(max_dd)) if max_dd < 0 else np.nan,
        "n_days": len(equity),
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {}
    pnl = trades["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_win = wins.sum()
    gross_loss = -losses.sum()

    return {
        "n_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else np.inf,
        "expectancy": pnl.mean(),
        "expectancy_pct": trades["return_pct"].mean() if "return_pct" in trades else np.nan,
        "best": pnl.max(),
        "worst": pnl.min(),
        "avg_hold_days": trades["hold_days"].mean() if "hold_days" in trades else np.nan,
    }


def summary(equity: pd.Series, trades: pd.DataFrame) -> dict:
    return {**equity_stats(equity), **trade_stats(trades)}
