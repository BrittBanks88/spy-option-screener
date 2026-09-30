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

SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]
WORKSHEET_NAME = "Tracked Alerts"
COLUMNS = ["id", "date_added", "contract", "entry_price", "gap_date",
           "gap_type", "target", "stop", "time_stop_date", "status",
           "resolution_date", "resolution_spy_price", "exit_price",
           "pnl_dollars", "pnl_pct", "notes"]


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


def add_entry(ws, *, contract: str, entry_price: float, date_added: str,
             gap_date: str, gap_type: str, target: float, stop: float,
             time_stop_date: str, notes: str = "") -> str:
    df = list_entries(ws)
    ids = pd.to_numeric(df["id"], errors="coerce").dropna()
    new_id = str(int(ids.max()) + 1) if len(ids) else "1"
    row = {c: "" for c in COLUMNS}
    row.update({"id": new_id, "date_added": date_added, "contract": contract,
               "entry_price": entry_price, "gap_date": gap_date,
               "gap_type": gap_type, "target": target, "stop": stop,
               "time_stop_date": time_stop_date, "status": "open",
               "notes": notes})
    ws.append_row([row[c] for c in COLUMNS])
    return new_id


def update_exit(ws, row_id: str, exit_price: float, exit_date: str) -> None:
    df = list_entries(ws)
    idx = df.index[df["id"].astype(str) == str(row_id)]
    if len(idx) == 0:
        raise ValueError(f"no tracked entry with id {row_id}")
    i = int(idx[0])
    entry_price = float(df.loc[i, "entry_price"])
    pnl_dollars = (exit_price - entry_price) * 100
    pnl_pct = (exit_price / entry_price - 1.0) if entry_price else 0.0
    sheet_row = i + 2  # +1 for header, +1 for 1-indexing
    ws.update(f"M{sheet_row}:O{sheet_row}", [[exit_price, pnl_dollars, pnl_pct]])
    if not str(df.loc[i, "resolution_date"]).strip():
        ws.update(f"K{sheet_row}:L{sheet_row}",
                 [[exit_date, float(df.loc[i, "resolution_spy_price"] or 0) or ""]])


def refresh_statuses(ws, daily: pd.DataFrame) -> int:
    """Check each OPEN row against real SPY price data since it was added:
    has it hit target, hit stop, or run past its time-stop date? Writes any
    resolved rows back to the sheet. Returns how many were updated."""
    df = list_entries(ws)
    if df.empty:
        return 0
    daily = daily.copy()
    daily.index = pd.DatetimeIndex(daily.index)
    today = daily.index[-1]
    updated = 0

    for i, row in df.iterrows():
        if row["status"] != "open":
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

        status, res_date, res_px = None, None, None
        for d, bar in path.iterrows():
            hit_target = (bar["low"] <= target) if gap_up else (bar["high"] >= target)
            hit_stop = (bar["high"] >= stop) if gap_up else (bar["low"] <= stop)
            if hit_target:
                status, res_date, res_px = "hit_target", d, bar["close"]
                break
            if hit_stop:
                status, res_date, res_px = "hit_stop", d, bar["close"]
                break
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
