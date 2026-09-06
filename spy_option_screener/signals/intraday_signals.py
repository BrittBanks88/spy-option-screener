"""Same-day intraday signals for long SPY weekly options.

Both signals are known by the ``signal_cutoff`` used to build the feature frame
(default 10:30 ET), so a trade opened at that time or later has no look-ahead.
Each returns a DataFrame indexed by session date with:

    direction   : +1 buy calls, -1 buy puts, 0 stand aside
    confidence  : 0..1
    edge_note   : why

Grounded in the intraday research (research/intraday_pattern.py):
  * Big opening gaps UP tend to continue (~+13 bps more, ~61% up); gap DOWNs
    have no reliable edge, so gap-down trading is opt-in and down-weighted.
  * ~70% of opening-range breakouts hold their direction into the close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _frame(index):
    return pd.DataFrame({
        "direction": pd.Series(0, index=index, dtype=int),
        "confidence": pd.Series(0.0, index=index, dtype=float),
        "edge_note": pd.Series("", index=index, dtype=object),
    })


def gap_continuation(feat: pd.DataFrame, up_threshold=0.003, strong=0.007,
                     trade_gap_down=False, down_threshold=0.006) -> pd.DataFrame:
    """Trade in the direction of a large opening gap."""
    out = _frame(feat.index)
    gap = feat["gap"]

    up = gap >= up_threshold
    conf_up = ((gap - up_threshold) / (strong - up_threshold)).clip(0, 1)
    out.loc[up, "direction"] = 1
    out.loc[up, "confidence"] = conf_up[up]
    out.loc[up, "edge_note"] = feat.loc[up, "gap"].map(
        lambda g: f"gap up {g:+.2%} — continuation")

    if trade_gap_down:
        dn = gap <= -down_threshold
        conf_dn = ((-gap - down_threshold) / (strong - down_threshold)).clip(0, 1)
        out.loc[dn, "direction"] = -1
        out.loc[dn, "confidence"] = 0.6 * conf_dn[dn]   # weaker edge historically
        out.loc[dn, "edge_note"] = feat.loc[dn, "gap"].map(
            lambda g: f"gap down {g:+.2%} — continuation (weak edge)")
    return out


def opening_range_breakout(feat: pd.DataFrame, max_break_minute=60,
                           max_or_width=0.006, align_gap=True,
                           direction_filter=0) -> pd.DataFrame:
    """Trade the first hold-beyond of the opening range.

    ``max_break_minute``: only take breaks within this many minutes of the OR
    window closing (a clean early break, not a 2pm lurch).
    ``max_or_width``: skip days whose opening range is already very wide.
    ``direction_filter``: +1 = only long-call breaks, -1 = only long-put, 0 = both.
    Note: this signal is under-validated on free data (needs <=30-min bars, of
    which yfinance only serves ~60 days). Keep its weight modest.
    """
    out = _frame(feat.index)
    d = feat["orb_dir"].fillna(0).astype(int)
    bm = feat["orb_break_minute"]
    width = feat["or_width"].fillna(1.0)

    ok = (d != 0) & (bm <= max_break_minute) & (width <= max_or_width)
    if direction_filter:
        ok = ok & (d == direction_filter)
    # cleaner range -> higher confidence; earlier break -> higher confidence
    conf = (0.4
            + 0.3 * (1 - (width / max_or_width)).clip(0, 1)
            + 0.3 * (1 - (bm / max_break_minute)).clip(0, 1))
    if align_gap:
        # boost when the gap agrees with the break, trim when it fights it
        agree = np.sign(feat["gap"].fillna(0)) == d
        conf = conf * np.where(agree, 1.0, 0.7)

    out.loc[ok, "direction"] = d[ok]
    out.loc[ok, "confidence"] = np.clip(conf[ok], 0, 1)
    out.loc[ok, "edge_note"] = d[ok].map(
        {1: "held above opening range", -1: "broke below opening range"})
    return out


SIGNALS = {
    "gap_continuation": gap_continuation,
    "opening_range_breakout": opening_range_breakout,
}


def combine_intraday(feat: pd.DataFrame, names=None, weights=None,
                     daily_bias: pd.DataFrame | None = None,
                     daily_bias_weight=0.5) -> pd.DataFrame:
    """Blend intraday signals (and optionally a daily directional bias).

    ``daily_bias`` is a {direction, confidence} frame from signals.strategies
    (e.g. the momentum/mean-reversion combo), reindexed onto ``feat`` dates.
    """
    names = names or list(SIGNALS)
    # gap-continuation is the better-evidenced edge; ORB is thin on free data
    default_w = {"gap_continuation": 1.0, "opening_range_breakout": 0.4}
    weights = weights or {n: default_w.get(n, 1.0) for n in names}
    parts = {n: SIGNALS[n](feat) for n in names}

    score = sum(weights[n] * parts[n]["direction"] * parts[n]["confidence"]
                for n in names)
    total_w = sum(abs(weights[n]) for n in names)

    if daily_bias is not None:
        db = daily_bias.reindex(feat.index).fillna({"direction": 0,
                                                    "confidence": 0.0})
        score = score + daily_bias_weight * db["direction"] * db["confidence"]
        total_w += abs(daily_bias_weight)

    out = _frame(feat.index)
    out["direction"] = np.sign(score).astype(int)
    out["confidence"] = (score.abs() / max(total_w, 1e-9)).clip(0, 1)
    for n in names:
        active = (parts[n]["direction"] == out["direction"]) & (parts[n]["direction"] != 0)
        blank = active & (out["edge_note"] == "")
        out.loc[blank, "edge_note"] = parts[n].loc[blank, "edge_note"]
    return out
