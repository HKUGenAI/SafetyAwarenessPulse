"""
Ingest Traditional Chinese PPT newspaper cuttings into a local SQLite events DB.

Each dated slide becomes one event row keyed by iso_date (YYYY-MM-DD).
Original wording is preserved; nothing is translated.

Usage:
    python document_ingest.py
    python document_ingest.py --reset
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Force UTF-8 console output so Traditional Chinese prints correctly on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from config import EVENTS_DB_PATH, PPT_DIR
from events_db import connect, init_db, insert_event, reset_db

# PPT files in this project use DD-MM-YYYY (e.g. 29-12-2023).
_DATE_DD_MM_YYYY = re.compile(
    r"(?P<day>0?[1-9]|[12]\d|3[01])[-/.](?P<month>0?[1-9]|1[0-2])[-/.](?P<year>19\d{2}|20\d{2})"
)
_DATE_YYYY_MM_DD = re.compile(
    r"(?P<year>19\d{2}|20\d{2})[-/.](?P<month>0?[1-9]|1[0-2])[-/.](?P<day>0?[1-9]|[12]\d|3[01])"
)
_DATE_CHINESE = re.compile(
    r"(?P<year>19\d{2}|20\d{2})\s*年\s*(?P<month>0?[1-9]|1[0-2])\s*月\s*(?P<day>0?[1-9]|[12]\d|3[01])\s*日?"
)
_FIELD_PATTERNS = {
    "event_date_raw": re.compile(r"日期\s*[:：]\s*([^\n]+)"),
    "location": re.compile(r"地點\s*[:：]\s*([^\n]+)"),
    "category": re.compile(r"分類\s*[:：]\s*([^\n]+)"),
}


def iter_shapes(shapes: Any) -> Any:
    """Yield shapes, including children inside groups."""
    for shape in shapes:
        yield shape
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from iter_shapes(shape.shapes)


def _paragraph_text(paragraph: Any) -> str:
    """Join runs so Chinese characters split across runs stay in original order."""
    parts: list[str] = []
    for run in paragraph.runs:
        parts.append(run.text or "")
    text = "".join(parts).strip()
    if not text and paragraph.text:
        text = paragraph.text.strip()
    return text


def extract_shape_text(shape: Any) -> list[str]:
    """Collect visible text from a shape, table, or notes-like text frame."""
    lines: list[str] = []
    if getattr(shape, "has_text_frame", False):
        for paragraph in shape.text_frame.paragraphs:
            line = _paragraph_text(paragraph)
            if line:
                lines.append(line)
    if getattr(shape, "has_table", False):
        for row in shape.table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
    return lines


def extract_slide_text(slide: Any) -> str:
    """Read all text from one slide. Original Traditional Chinese is kept unchanged."""
    lines: list[str] = []
    for shape in iter_shapes(slide.shapes):
        lines.extend(extract_shape_text(shape))

    notes_slide = getattr(slide, "notes_slide", None)
    if notes_slide is not None and notes_slide.notes_text_frame is not None:
        notes = notes_slide.notes_text_frame.text.strip()
        if notes:
            lines.append(notes)

    cleaned = "\n".join(line.strip() for line in lines if line.strip())
    return cleaned.strip()


def _pad_month_day(month: str, day: str) -> str:
    return f"{int(month):02d}-{int(day):02d}"


def parse_event_date(text: str) -> tuple[str, str]:
    """
    Parse an accident date from slide text.

    Returns:
        iso_date: YYYY-MM-DD or empty string
        month_day: MM-DD or empty string (used for same-day-in-history lookup)
    """
    preferred = ""
    date_field = _FIELD_PATTERNS["event_date_raw"].search(text)
    if date_field:
        preferred = date_field.group(1)

    candidates = [preferred, text] if preferred else [text]
    for blob in candidates:
        # Prefer DD-MM-YYYY because that is the format used in these PPTs.
        match = _DATE_DD_MM_YYYY.search(blob)
        if match:
            iso = datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            ).strftime("%Y-%m-%d")
            return iso, _pad_month_day(match.group("month"), match.group("day"))

        match = _DATE_CHINESE.search(blob)
        if match:
            iso = datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            ).strftime("%Y-%m-%d")
            return iso, _pad_month_day(match.group("month"), match.group("day"))

        match = _DATE_YYYY_MM_DD.search(blob)
        if match:
            iso = datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            ).strftime("%Y-%m-%d")
            return iso, _pad_month_day(match.group("month"), match.group("day"))

    return "", ""


def parse_slide_fields(text: str) -> dict[str, str]:
    """Extract structured fields from a newspaper-cutting slide."""
    fields: dict[str, str] = {}
    for key, pattern in _FIELD_PATTERNS.items():
        match = pattern.search(text)
        fields[key] = match.group(1).strip() if match else ""

    title = text.splitlines()[0].strip() if text else ""
    iso_date, month_day = parse_event_date(text)
    fields.update(
        {
            "title": title[:200],
            "iso_date": iso_date,
            "month_day": month_day,
        }
    )
    return fields


def load_ppt_events(ppt_dir: Path) -> tuple[list[dict[str, Any]], int]:
    """
    Load every .pptx/.ppt file and turn each dated slide into an event dict.

    Slides without a parseable date are skipped (iso_date is the primary key).
    Returns (events, skipped_undated_count).
    """
    ppt_files = sorted(ppt_dir.glob("*.pptx")) + sorted(ppt_dir.glob("*.ppt"))
    ppt_files = [path for path in ppt_files if not path.name.startswith("~$")]
    if not ppt_files:
        raise FileNotFoundError(
            f"No PPT files found in {ppt_dir}. Place the 4 Traditional Chinese PPTs there first."
        )

    events: list[dict[str, Any]] = []
    skipped_undated = 0
    for ppt_path in ppt_files:
        print(f"[ingest] Opening {ppt_path.name} ...")
        try:
            presentation = Presentation(str(ppt_path))
        except Exception as exc:  # noqa: BLE001 - keep ingest running for other files
            print(f"[ingest] ERROR: failed to open {ppt_path.name}: {exc}")
            continue

        kept = 0
        for index, slide in enumerate(presentation.slides, start=1):
            text = extract_slide_text(slide)
            # Skip nearly empty cover / divider slides.
            if len(text) < 20:
                continue

            fields = parse_slide_fields(text)
            if not fields["iso_date"]:
                skipped_undated += 1
                continue

            events.append(
                {
                    "iso_date": fields["iso_date"],
                    "month_day": fields["month_day"],
                    "title": fields["title"],
                    "location": fields["location"][:200],
                    "category": fields["category"][:200],
                    "content": text,
                    "source_file": ppt_path.name,
                    "slide_number": index,
                    "language": "zh-Hant",
                }
            )
            kept += 1

        print(f"[ingest] {ppt_path.name}: {kept} dated slides / {len(presentation.slides)} total")

    print(f"[ingest] Total dated events: {len(events)}")
    print(f"[ingest] Skipped undated (non-empty) slides: {skipped_undated}")
    return events, skipped_undated


def persist_to_sqlite(events: list[dict[str, Any]], reset: bool = False) -> None:
    """Write structured events into SQLite. One row per accident (same day kept separate)."""
    if not events:
        raise ValueError("No dated events to store. Check PPT date parsing output.")

    if reset:
        reset_db(EVENTS_DB_PATH)
        print(f"[ingest] Reset: deleted {EVENTS_DB_PATH}")

    inserted = 0
    replaced = 0
    with connect(EVENTS_DB_PATH) as conn:
        init_db(conn)
        for event in events:
            result = insert_event(conn, event)
            if result == "inserted":
                inserted += 1
            elif result == "replaced":
                replaced += 1
        conn.commit()
        total = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        same_day = conn.execute(
            """
            SELECT COUNT(*) AS n FROM (
                SELECT iso_date FROM events GROUP BY iso_date HAVING COUNT(*) > 1
            )
            """
        ).fetchone()["n"]

    print(f"[ingest] Done. Saved to {EVENTS_DB_PATH}")
    print(
        f"[ingest] Rows inserted: {inserted}, replaced (same slide): {replaced}, "
        f"total rows: {total}, dates with multiple accidents: {same_day}"
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PPT files into local SQLite events DB.")
    parser.add_argument(
        "--ppt-dir",
        type=Path,
        default=PPT_DIR,
        help="Folder that contains the 4 Traditional Chinese PPT files.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing SQLite database before ingesting.",
    )
    args = parser.parse_args()

    print(f"[ingest] PPT folder: {args.ppt_dir}")
    print(f"[ingest] SQLite DB: {EVENTS_DB_PATH}")
    events, _skipped = load_ppt_events(args.ppt_dir)
    persist_to_sqlite(events, reset=args.reset)


if __name__ == "__main__":
    main()
