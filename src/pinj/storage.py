"""SQLite persistence for full per-run auditability."""
from __future__ import annotations
import json, sqlite3
from contextlib import contextmanager


@contextmanager
def db(path: str = "results.sqlite"):
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS runs ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, payload TEXT)"
    )
    try:
        yield con
        con.commit()
    finally:
        con.close()


def save(con, kind: str, obj: dict) -> None:
    con.execute("INSERT INTO runs (kind, payload) VALUES (?, ?)",
                (kind, json.dumps(obj)))
