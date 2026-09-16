"""
Ingest Traditional Chinese PPT newspaper cuttings into a local ChromaDB store.

Do not translate PPT content. Original wording is preserved in chunks and embeddings.

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

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PPT_DIR,
)

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


def build_embeddings() -> HuggingFaceEmbeddings:
    """Create a local multilingual embedding model (no cloud API)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


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

    # Collapse extra blank lines but keep paragraph boundaries.
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


def load_ppt_documents(ppt_dir: Path) -> list[Document]:
    """Load every .pptx/.ppt file and turn each slide into a LangChain Document."""
    ppt_files = sorted(ppt_dir.glob("*.pptx")) + sorted(ppt_dir.glob("*.ppt"))
    ppt_files = [path for path in ppt_files if not path.name.startswith("~$")]
    if not ppt_files:
        raise FileNotFoundError(
            f"No PPT files found in {ppt_dir}. Place the 4 Traditional Chinese PPTs there first."
        )

    documents: list[Document] = []
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
            metadata = {
                "source_file": ppt_path.name,
                "source_path": str(ppt_path),
                "slide_number": index,
                "title": fields["title"],
                "iso_date": fields["iso_date"],
                "month_day": fields["month_day"],
                "location": fields["location"][:200],
                "category": fields["category"][:200],
                "language": "zh-Hant",
            }
            documents.append(Document(page_content=text, metadata=metadata))
            kept += 1

        print(f"[ingest] {ppt_path.name}: {kept} non-empty slides / {len(presentation.slides)} total")

    print(f"[ingest] Total non-empty slides: {len(documents)}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split long slides, but keep a typical accident record as one chunk.

    Separators include Chinese punctuation so Traditional Chinese is not broken mid-word.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documents:
        if len(doc.page_content) <= CHUNK_SIZE:
            chunks.append(doc)
            continue
        for part in splitter.split_text(doc.page_content):
            chunks.append(Document(page_content=part, metadata=dict(doc.metadata)))
    return chunks


def persist_to_chroma(chunks: list[Document], reset: bool = False) -> None:
    """Embed original Traditional Chinese chunks and save them to local ChromaDB."""
    if not chunks:
        raise ValueError("No text chunks to embed. Check PPT parsing output.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = build_embeddings()

    if reset and CHROMA_DIR.exists():
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        try:
            vector_store.delete_collection()
            print("[ingest] Existing Chroma collection deleted.")
        except Exception as exc:  # noqa: BLE001
            print(f"[ingest] WARNING: could not delete old collection: {exc}")

    print(f"[ingest] Embedding {len(chunks)} chunks with {EMBEDDING_MODEL_NAME} ...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    dated = sum(1 for chunk in chunks if chunk.metadata.get("month_day"))
    print(f"[ingest] Done. Saved to {CHROMA_DIR}")
    print(f"[ingest] Chunks with parsed MM-DD metadata: {dated}/{len(chunks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PPT files into local ChromaDB.")
    parser.add_argument(
        "--ppt-dir",
        type=Path,
        default=PPT_DIR,
        help="Folder that contains the 4 Traditional Chinese PPT files.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma collection before ingesting.",
    )
    args = parser.parse_args()

    print(f"[ingest] PPT folder: {args.ppt_dir}")
    documents = load_ppt_documents(args.ppt_dir)
    chunks = split_documents(documents)
    print(f"[ingest] Chunks after split: {len(chunks)}")
    persist_to_chroma(chunks, reset=args.reset)


if __name__ == "__main__":
    main()
