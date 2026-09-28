"""
SQLite store for structured workplace-accident events.

Each accident is one row (same calendar day may have many rows).
Lookup for daily alerts uses month_day (MM-DD); the alert path picks one at random.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Any

from config import EVENTS_DB_PATH

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    iso_date     TEXT NOT NULL,     -- YYYY-MM-DD (event calendar date; not unique)
    month_day    TEXT NOT NULL,     -- MM-DD for historical same-day search
    title        TEXT NOT NULL DEFAULT '',
    location     TEXT NOT NULL DEFAULT '',
    category     TEXT NOT NULL DEFAULT '',
    content      TEXT NOT NULL,     -- original Traditional Chinese PPT text
    source_file  TEXT NOT NULL DEFAULT '',
    slide_number INTEGER NOT NULL DEFAULT 0,
    language     TEXT NOT NULL DEFAULT 'zh-Hant',
    UNIQUE (source_file, slide_number)
);

CREATE INDEX IF NOT EXISTS idx_events_month_day ON events(month_day);
CREATE INDEX IF NOT EXISTS idx_events_iso_date ON events(iso_date);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else EVENTS_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def reset_db(db_path: Path | None = None) -> None:
    """Delete the SQLite file (if present) so the next ingest starts clean."""
    path = Path(db_path) if db_path is not None else EVENTS_DB_PATH
    if path.exists():
        path.unlink()


def db_ready(db_path: Path | None = None) -> bool:
    path = Path(db_path) if db_path is not None else EVENTS_DB_PATH
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with connect(path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            if not row or int(row["n"]) == 0:
                return False
            count = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
            return bool(count and int(count["n"]) > 0)
    except sqlite3.Error:
        return False


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]) if "id" in row.keys() else 0,
        "iso_date": row["iso_date"] or "",
        "month_day": row["month_day"] or "",
        "title": row["title"] or "",
        "location": row["location"] or "",
        "category": row["category"] or "",
        "content": row["content"] or "",
        "source_file": row["source_file"] or "",
        "slide_number": int(row["slide_number"] or 0),
        "language": row["language"] or "zh-Hant",
    }


def insert_event(conn: sqlite3.Connection, event: dict[str, Any]) -> str:
    """
    Insert one accident as its own row (same iso_date allowed many times).

    Dedup key is (source_file, slide_number) so re-ingest without --reset
    does not create duplicates from the same slide.
    Returns: 'inserted' | 'replaced' | 'skipped'
    """
    iso_date = (event.get("iso_date") or "").strip()
    if not iso_date:
        return "skipped"

    month_day = (event.get("month_day") or "").strip()
    if not month_day and len(iso_date) >= 10:
        month_day = iso_date[5:10]

    content = (event.get("content") or "").strip()
    if not content:
        return "skipped"

    title = (event.get("title") or "")[:200]
    location = (event.get("location") or "")[:200]
    category = (event.get("category") or "")[:200]
    source_file = event.get("source_file") or ""
    slide_number = int(event.get("slide_number") or 0)
    language = event.get("language") or "zh-Hant"

    existing = conn.execute(
        "SELECT id FROM events WHERE source_file = ? AND slide_number = ?",
        (source_file, slide_number),
    ).fetchone()

    if existing is None:
        conn.execute(
            """
            INSERT INTO events (
                iso_date, month_day, title, location, category,
                content, source_file, slide_number, language
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                iso_date,
                month_day,
                title,
                location,
                category,
                content,
                source_file,
                slide_number,
                language,
            ),
        )
        return "inserted"

    conn.execute(
        """
        UPDATE events SET
            iso_date = ?,
            month_day = ?,
            title = ?,
            location = ?,
            category = ?,
            content = ?,
            language = ?
        WHERE id = ?
        """,
        (
            iso_date,
            month_day,
            title,
            location,
            category,
            content,
            language,
            int(existing["id"]),
        ),
    )
    return "replaced"


def fetch_by_month_day(
    month_day: str,
    *,
    extra_query: str = "",
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return all events whose month_day matches MM-DD, ordered by iso_date then id."""
    month_day = (month_day or "").strip()
    if not month_day:
        return []

    extra = (extra_query or "").strip()
    with connect(db_path) as conn:
        init_db(conn)
        if extra:
            like = f"%{extra}%"
            rows = conn.execute(
                """
                SELECT * FROM events
                WHERE month_day = ?
                  AND (content LIKE ? OR title LIKE ? OR location LIKE ? OR category LIKE ?)
                ORDER BY iso_date, id
                """,
                (month_day, like, like, like, like),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM events WHERE month_day = ? ORDER BY iso_date, id",
                (month_day,),
            ).fetchall()
    return [_row_to_event(row) for row in rows]


def fetch_one_random_by_month_day(
    month_day: str,
    *,
    extra_query: str = "",
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    """Pick one random event for MM-DD (daily alert uses a single accident)."""
    events = fetch_by_month_day(month_day, extra_query=extra_query, db_path=db_path)
    if not events:
        return None
    return random.choice(events)


def fetch_by_iso_date(
    iso_date: str,
    *,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return all events on YYYY-MM-DD (may be more than one)."""
    iso_date = (iso_date or "").strip()
    if not iso_date:
        return []
    with connect(db_path) as conn:
        init_db(conn)
        rows = conn.execute(
            "SELECT * FROM events WHERE iso_date = ? ORDER BY id",
            (iso_date,),
        ).fetchall()
    return [_row_to_event(row) for row in rows]


def count_events(db_path: Path | None = None) -> int:
    if not db_ready(db_path):
        return 0
    with connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
    return int(row["n"]) if row else 0
