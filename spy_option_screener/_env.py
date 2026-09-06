"""Zero-dependency .env loader. Reads KEY=VALUE lines from a .env file at the
project root into os.environ (without overwriting anything already set).
"""
from __future__ import annotations

import os
from pathlib import Path


def load(path: str | Path | None = None) -> None:
    p = Path(path) if path else Path(__file__).resolve().parents[1] / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)
