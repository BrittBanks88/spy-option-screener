"""Google Sheets-backed store for tracked SPY gap-fill alerts (accuracy log).

One row per tracked contract. Target/stop/time-out are checked automatically
against real SPY daily price data (refresh_statuses) -- that's what tests the
SCREENER's accuracy. entry_price/exit_price/pnl come from the user's own
brokerage screenshots, not a reconstructed estimate -- that's what tests
their REAL trade result. Two different, deliberately separate numbers.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

from ..signals.gaps import first_resolution

SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]
WORKSHEET_NAME = "Tracked Alerts"
COLUMNS = ["id", "date_added", "contract", "entry_price", "gap_date",
           "gap_type", "target", "stop", "time_stop_date", "status",
           "resolution_date", "resolution_spy_price", "exit_price",
           "pnl_dollars", "pnl_pct", "notes", "exit_date"]


# ---- credential / connection helpers --------------------------------------
def client_from_info(creds_info: dict) -> gspread.Client:
    """Build a client from a dict (e.g. st.secrets['gcp_service_account'])."""
    creds = Credentials.from_service_account_info(dict(creds_info), scopes=SCOPES)
    return gspread.authorize(creds)


def client_from_local_toml(path: str = ".streamlit/secrets.toml") -> tuple[gspread.Client, str]:
    """For use outside a running Streamlit app (e.g. adding an entry from a
    screenshot via a one-off script). Returns (client, sheet_url)."""
    import tomllib
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / path
    with open(p, "rb") as f:
        secrets = tomllib.load(f)
    client = client_from_info(secrets["gcp_service_account"])
    return client, secrets["gsheet"]["url"]


def get_worksheet(client: gspread.Client, sheet_url: str):
    sh = client.open_by_url(sheet_url)
    try:
        ws = sh.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(WORKSHEET_NAME, rows=200, cols=len(COLUMNS))
        ws.append_row(COLUMNS)
        return ws
    if ws.row_values(1) != COLUMNS:
        ws.update("A1", [COLUMNS])
    return ws


# ---- reads / writes ---------------------------------------------------------
def list_entries(ws) -> pd.DataFrame:
    records = ws.get_all_records()
    if not records:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(records)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[COLUMNS]


PENDING_STATUSES = ("open", "watching")
TERMINAL_STATUSES = ("hit_target", "hit_stop", "timed_out")


def trade_result(exit_price, pnl_dollars) -> str | None:
    """'win' / 'loss' / 'flat' for a sold position, None if not sold yet.
    Judged on the user's real P&L only -- independent of the screener's
    signal status (which is what the accuracy score is built from)."""
    ex = pd.to_numeric(exit_price, errors="coerce")
    pnl = pd.to_numeric(pnl_dollars, errors="coerce")
    if pd.isna(ex) or pd.isna(pnl):
        return None
    return "win" if pnl > 0 else "loss" if pnl < 0 else "flat"


def signal_accuracy(df: pd.DataFrame) -> tuple[int, int]:
    """(signals that hit target, resolved signals) -- counted per DISTINCT gap
    signal, not per row. Two contracts tracking the same gap are one signal;
    counting rows would double-weight it and inflate the score."""
    resolved = df[~df["status"].isin(PENDING_STATUSES)]
    signals = resolved.drop_duplicates(subset=["gap_date", "gap_type"])
    return int((signals["status"] == "hit_target").sum()), len(signals)


def add_entry(ws, *, contract: str, entry_price: float, date_added: str,
             gap_date: str, gap_type: str, target: float, stop: float,
             time_stop_date: str, notes: str = "", status: str = "open") -> str:
    if status not in PENDING_STATUSES:
        raise ValueError(f"status must be one of {PENDING_STATUSES}")
    df = list_entries(ws)
    ids = pd.to_numeric(df["id"], errors="coerce").dropna()
    new_id = str(int(ids.max()) + 1) if len(ids) else "1"
    row = {c: "" for c in COLUMNS}
    row.update({"id": new_id, "date_added": date_added, "contract": contract,
               "entry_price": entry_price, "gap_date": gap_date,
               "gap_type": gap_type, "target": target, "stop": stop,
               "time_stop_date": time_stop_date, "status": status,
               "notes": notes})
    ws.append_row([row[c] for c in COLUMNS])
    return new_id


def update_exit(ws, row_id: str, exit_price: float, exit_date: str,
               entry_price: float | None = None, notes: str | None = None) -> None:
    """Record a sale. exit_date goes in its own column -- it is NOT the
    signal's resolution date (that's set only by refresh_statuses from real
    price data), so selling early never overwrites when the signal resolved.
    Pass entry_price to correct the cost basis to the broker's own figure
    before P&L is computed."""
    df = list_entries(ws)
    idx = df.index[df["id"].astype(str) == str(row_id)]
    if len(idx) == 0:
        raise ValueError(f"no tracked entry with id {row_id}")
    i = int(idx[0])
    sheet_row = i + 2  # +1 for header, +1 for 1-indexing
    if entry_price is not None:
        ws.update(f"D{sheet_row}", [[entry_price]])
    else:
        entry_price = float(df.loc[i, "entry_price"])
    pnl_dollars = round((exit_price - entry_price) * 100, 2)
    pnl_pct = round(exit_price / entry_price - 1.0, 6) if entry_price else 0.0
    ws.update(f"M{sheet_row}:O{sheet_row}", [[exit_price, pnl_dollars, pnl_pct]])
    ws.update(f"Q{sheet_row}", [[exit_date]])
    if notes is not None:
        ws.update(f"P{sheet_row}", [[notes]])


def refresh_statuses(ws, daily: pd.DataFrame) -> int:
    """Check each pending row (open OR watching) against real SPY price data
    since it was added: has it hit target, hit stop, or run past its
    time-stop date? Writes any resolved rows back to the sheet. Returns how
    many were updated."""
    df = list_entries(ws)
    if df.empty:
        return 0
    daily = daily.copy()
    daily.index = pd.DatetimeIndex(daily.index)
    today = daily.index[-1]
    updated = 0

    for i, row in df.iterrows():
        if row["status"] not in PENDING_STATUSES:
            continue
        try:
            entry_date = pd.Timestamp(row["date_added"])
            target, stop = float(row["target"]), float(row["stop"])
            gap_up = row["gap_type"] == "up"
        except (ValueError, TypeError):
            continue
        path = daily.loc[daily.index >= entry_date]
        if path.empty:
            continue

        # same rule as the live screener and the backtest (see signals/gaps.py)
        outcome, res_ts = first_resolution(path, 1 if gap_up else -1, target, stop)
        status, res_date, res_px = None, None, None
        if outcome is not None:
            status = "hit_target" if outcome == "target" else "hit_stop"
            res_date, res_px = res_ts, path.loc[res_ts, "close"]
        if status is None:
            time_stop = pd.Timestamp(row["time_stop_date"])
            if today > time_stop:
                status, res_date = "timed_out", time_stop
                res_px = float(daily.loc[daily.index <= time_stop, "close"].iloc[-1])

        if status is not None:
            sheet_row = i + 2
            ws.update(f"J{sheet_row}:L{sheet_row}",
                     [[status, str(res_date.date()), float(res_px)]])
            updated += 1
    return updated
