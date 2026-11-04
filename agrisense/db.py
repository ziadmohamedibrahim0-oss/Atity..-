"""SQLite persistence for saved simulations and disease diagnoses."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime

DB_PATH = "agrisense.db"
_TABLES = ("simulations", "diagnoses")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS simulations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, field TEXT, strategy TEXT, plant_date TEXT,
    soil TEXT, system TEXT, irrigation_mm REAL, etc_mm REAL, eta_mm REAL, relative_yield_pct REAL, summary TEXT);
CREATE TABLE IF NOT EXISTS diagnoses (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, field TEXT, filename TEXT,
    label TEXT, confidence REAL, severity_pct REAL, model TEXT);
"""


def _connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _now():
    return datetime.now().isoformat(timespec="seconds")


def save_simulation(field, strategy, plant_date, soil, system, summary, path=DB_PATH):
    """Store one simulation summary (the dict returned by irrigation.summarize)."""
    with closing(_connect(path)) as conn, conn:
        conn.execute("INSERT INTO simulations (created_at, field, strategy, plant_date, soil, system, irrigation_mm,"
                     " etc_mm, eta_mm, relative_yield_pct, summary) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (_now(), field, strategy, str(plant_date), soil, system, summary["irrigation_mm"],
                      summary["etc_mm"], summary["eta_mm"], summary["relative_yield_pct"],
                      json.dumps(summary, default=float)))


def save_diagnosis(field, filename, label, confidence, severity_pct, model, path=DB_PATH):
    with closing(_connect(path)) as conn, conn:
        conn.execute("INSERT INTO diagnoses (created_at, field, filename, label, confidence, severity_pct, model)"
                     " VALUES (?,?,?,?,?,?,?)",
                     (_now(), field, filename, label, float(confidence), float(severity_pct), model))


def recent(table, field=None, limit=20, path=DB_PATH):
    """Latest rows of 'simulations' or 'diagnoses' (optionally for one field), newest first."""
    if table not in _TABLES:
        raise ValueError(f"table must be one of {_TABLES}")
    sql, args = f"SELECT * FROM {table}", []
    if field:
        sql, args = sql + " WHERE field = ?", [field]
    with closing(_connect(path)) as conn:
        return [dict(r) for r in conn.execute(sql + " ORDER BY id DESC LIMIT ?", args + [limit])]
