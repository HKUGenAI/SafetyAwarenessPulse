"""
Daily workplace safety-alert agent.

LLM: Azure OpenAI gpt-5.4-mini via LangChain AzureChatOpenAI (function calling / tool use).
Tools:
  A) search_local_work_accidents — SQLite lookup by date over the 4 Traditional Chinese PPTs
  B) search_labour_department — Hong Kong Labour Department press releases
     (https://www.labour.gov.hk/tc/major/content.php)
  C) search_web — Serper Google Search API for wider web lookup

Priority (stop at the first hit):
  1. Local SQLite: working accidents on the same MM-DD in previous years
  2. Labour Department news: https://www.labour.gov.hk/tc/major/content.php
  3. Web: working accidents around the world on this day
  4. Web: interesting historical facts on this day

All user-facing answers are Traditional Chinese. PPT source text is never translated.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain.tools import tool
from langchain_openai import AzureChatOpenAI

try:
    from langchain.agents import create_agent as _create_agent
except ImportError:  # langchain < 1.0 fallback
    from langgraph.prebuilt import create_react_agent as _create_agent

from config import (
    ACCIDENT_KEYWORDS,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    EVENTS_DB_PATH,
    HONG_KONG_TZ,
    LABOUR_DEPT_DOMAINS,
    LABOUR_DEPT_NEWS_URL,
)
from document_ingest import parse_event_date
from events_db import db_ready, fetch_by_month_day
from multi_search_api import SmartSearchTool

_MONTH_DAY_RE = re.compile(r"^(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

_serper: SmartSearchTool | None = None
_agent = None


class KnowledgeBaseError(RuntimeError):
    """Raised when the local SQLite events DB is missing or unreadable."""


def today_in_hong_kong(override: str | None = None) -> date:
    """Return today's date in Hong Kong, or a parsed override (YYYY-MM-DD)."""
    if override:
        override = override.strip()
        iso_match = _ISO_DATE_RE.match(override)
        if iso_match:
            return date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
        month_day = normalize_month_day(override)
        hk_today = datetime.now(ZoneInfo(HONG_KONG_TZ)).date()
        return date(hk_today.year, int(month_day[:2]), int(month_day[3:]))
    return datetime.now(ZoneInfo(HONG_KONG_TZ)).date()


def to_month_day(value: date) -> str:
    return value.strftime("%m-%d")


def normalize_month_day(raw: str) -> str:
    """Accept MM-DD, M-D, DD-MM (when unambiguous), or YYYY-MM-DD and return MM-DD."""
    text = (raw or "").strip()
    iso_match = _ISO_DATE_RE.match(text)
    if iso_match:
        return f"{iso_match.group(2)}-{iso_match.group(3)}"

    if _MONTH_DAY_RE.match(text):
        return text

    parsed_iso, month_day = parse_event_date(text)
    if month_day:
        return month_day
    if parsed_iso:
        return parsed_iso[5:]

    chinese = re.fullmatch(
        r"(0?[1-9]|1[0-2])\s*月\s*(0?[1-9]|[12]\d|3[01])\s*[日號]?",
        text,
    )
    if chinese:
        return f"{int(chinese.group(1)):02d}-{int(chinese.group(2)):02d}"

    loose = re.match(r"^(\d{1,2})[-/.](\d{1,2})$", text)
    if loose:
        left, right = int(loose.group(1)), int(loose.group(2))
        # If the left number cannot be a month, treat as DD-MM.
        if left > 12 and 1 <= right <= 12:
            return f"{right:02d}-{left:02d}"
        if 1 <= left <= 12 and 1 <= right <= 31:
            return f"{left:02d}-{right:02d}"

    raise ValueError(f"Cannot parse month-day from: {raw!r}. Expected MM-DD or YYYY-MM-DD.")


_EN_MONTHS = (
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_EN_MONTHS_SHORT = (
    "",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
_EN_MONTH_LOOKUP = {}
for _idx, _name in enumerate(_EN_MONTHS):
    if _name:
        _EN_MONTH_LOOKUP[_name.lower()] = _idx
        _EN_MONTH_LOOKUP[_EN_MONTHS_SHORT[_idx].lower()] = _idx
_EN_MONTH_LOOKUP["sept"] = 9

# Two-digit alternatives first so 9月16日 is day=16, not day=1.
# Never treat a bare day (16 / 17) as a date — that matches ages, headcounts, 2016.
_ISO_DATE_IN_TEXT = re.compile(
    r"(?:19|20)\d{2}[-/.](?P<month>1[0-2]|0?[1-9])[-/.](?P<day>[12]\d|3[01]|0?[1-9])"
)
_DMY_DATE_IN_TEXT = re.compile(
    r"(?P<day>[12]\d|3[01]|0?[1-9])[-/.](?P<month>1[0-2]|0?[1-9])[-/.](?:19|20)\d{2}"
)
_MDY_DATE_IN_TEXT = re.compile(
    r"(?P<month>1[0-2]|0?[1-9])[-/.](?P<day>[12]\d|3[01]|0?[1-9])[-/.](?:19|20)\d{2}"
)
_CHINESE_YMD_IN_TEXT = re.compile(
    r"(?:(?:19|20)\d{2}\s*年\s*)?(?P<month>1[0-2]|0?[1-9])\s*月\s*"
    r"(?P<day>[12]\d|3[01]|0?[1-9])\s*[日號]?(?!\d)"
)
_EN_MONTH_DAY_IN_TEXT = re.compile(
    r"(?P<month>January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?\s+"
    r"(?P<day>[12]\d|3[01]|0?[1-9])(?:st|nd|rd|th)?",
    re.I,
)
_EN_DAY_MONTH_IN_TEXT = re.compile(
    r"(?P<day>[12]\d|3[01]|0?[1-9])(?:st|nd|rd|th)?\s+(?:of\s+)?"
    r"(?P<month>January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?",
    re.I,
)
_YEARLESS_NUMERIC_DATE = re.compile(
    r"(?<!\d)(?P<a>[12]\d|3[01]|0?[1-9])(?P<sep>[./-])(?P<b>[12]\d|3[01]|0?[1-9])(?!\d)"
)


def _day_ordinal(day: int) -> str:
    if 10 < day < 14:
        return f"{day}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def _unique_longest_first(items: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in sorted(items, key=len, reverse=True):
        if item and item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def date_keyword_variants(month_day: str, *, with_years: bool = True) -> list[str]:
    """Every common way the same MM-DD may be written. Never a bare day or month digit."""
    month_day = normalize_month_day(month_day)
    month = int(month_day[:2])
    day = int(month_day[3:])
    mm, dd = f"{month:02d}", f"{day:02d}"
    m, d = str(month), str(day)
    en = _EN_MONTHS[month]
    en_short = _EN_MONTHS_SHORT[month]
    ordinal = _day_ordinal(day)
    variants = [
        f"{mm}-{dd}",
        f"{m}-{d}",
        f"{mm}/{dd}",
        f"{m}/{d}",
        f"{mm}.{dd}",
        f"{m}.{d}",
        f"{dd}-{mm}",
        f"{d}-{m}",
        f"{dd}/{mm}",
        f"{d}/{m}",
        f"{dd}.{mm}",
        f"{d}.{m}",
        f"{m}月{d}日",
        f"{mm}月{dd}日",
        f"{m}月{d}號",
        f"{mm}月{dd}號",
        f"{m} 月 {d} 日",
        f"{mm} 月 {dd} 日",
        f"{m}月{d}",
        f"{d}日{m}月",
        f"{dd}日{mm}月",
        f"{en} {day}",
        f"{en} {dd}",
        f"{en} {ordinal}",
        f"{en_short} {day}",
        f"{en_short} {dd}",
        f"{en_short}. {day}",
        f"{en_short}. {ordinal}",
        f"{day} {en}",
        f"{dd} {en}",
        f"{ordinal} {en}",
        f"{ordinal} of {en}",
        f"{day} {en_short}",
        f"{dd} {en_short}",
        f"{ordinal} {en_short}",
    ]
    if with_years:
        for year in range(2018, 2028):
            variants.extend(
                [
                    f"{year}-{mm}-{dd}",
                    f"{year}/{mm}/{dd}",
                    f"{year}.{mm}.{dd}",
                    f"{dd}-{mm}-{year}",
                    f"{dd}/{mm}/{year}",
                    f"{mm}-{dd}-{year}",
                    f"{year}年{m}月{d}日",
                    f"{year}年{mm}月{dd}日",
                    f"{en} {day}, {year}",
                    f"{ordinal} {en} {year}",
                ]
            )
    return _unique_longest_first(variants)


def compact_date_keywords(month_day: str) -> list[str]:
    """Short, high-signal date spellings for Serper / local keyword queries."""
    month_day = normalize_month_day(month_day)
    month = int(month_day[:2])
    day = int(month_day[3:])
    return [
        f"{month}月{day}日",
        month_day,
        f"{month}.{day}",
        f"{month}/{day}",
        f"{day:02d}-{month:02d}",
        f"{_EN_MONTHS[month]} {day}",
        f"{day} {_EN_MONTHS[month]}",
        f"{_EN_MONTHS_SHORT[month]} {day}",
        f"{_day_ordinal(day)} {_EN_MONTHS[month]}",
    ]


def month_day_search_terms(month_day: str) -> list[str]:
    """Backward-compatible alias used by existing call sites."""
    return date_keyword_variants(month_day)


def _pad_md(month: str | int, day: str | int) -> str:
    return f"{int(month):02d}-{int(day):02d}"


def _add_yearless_numeric_dates(text: str, found: set[str]) -> None:
    """Match 9.16 / 09-16 / 16/09 only when not part of a larger number."""
    for match in _YEARLESS_NUMERIC_DATE.finditer(text):
        left, right = int(match.group("a")), int(match.group("b"))
        if 1 <= left <= 12 and 1 <= right <= 31:
            found.add(_pad_md(left, right))
        if 1 <= right <= 12 and 1 <= left <= 31:
            found.add(_pad_md(right, left))


def extract_incident_month_days(text: str) -> set[str]:
    """
    Pull real calendar dates out of incident text.

    Only structured dates count. A lone 17 (age, headcount, 2017) is ignored.
    9.16 / 09-16 / 16-09-2023 / 9月16日 / September 16 are accepted.
    9.17 does not satisfy a 09-16 query.
    """
    if not text:
        return set()
    found: set[str] = set()

    for match in _ISO_DATE_IN_TEXT.finditer(text):
        found.add(_pad_md(match.group("month"), match.group("day")))

    for match in _DMY_DATE_IN_TEXT.finditer(text):
        found.add(_pad_md(match.group("month"), match.group("day")))

    for match in _MDY_DATE_IN_TEXT.finditer(text):
        month, day = int(match.group("month")), int(match.group("day"))
        if day > 12:
            found.add(_pad_md(month, day))
        elif month == day:
            found.add(_pad_md(month, day))
        else:
            found.add(_pad_md(month, day))
            found.add(_pad_md(day, month))

    for match in _CHINESE_YMD_IN_TEXT.finditer(text):
        found.add(_pad_md(match.group("month"), match.group("day")))

    for match in _EN_MONTH_DAY_IN_TEXT.finditer(text):
        month = _EN_MONTH_LOOKUP.get(match.group("month").lower().rstrip("."))
        if month:
            found.add(_pad_md(month, match.group("day")))

    for match in _EN_DAY_MONTH_IN_TEXT.finditer(text):
        month = _EN_MONTH_LOOKUP.get(match.group("month").lower().rstrip("."))
        if month:
            found.add(_pad_md(month, match.group("day")))

    _add_yearless_numeric_dates(text, found)
    return found


def passes_same_date_guardrail(
    text: str,
    target_month_day: str,
    metadata_month_day: str = "",
) -> bool:
    """
    Keep an incident only if its event date is the requested MM-DD.

    Metadata from PPT ingest wins when present. Otherwise the body must contain
    an explicit date that maps to the same month-day. Nearby dates (09-17 when
    asking 09-16) are rejected even if the text mentions 16 or 17 as a count.
    """
    try:
        target = normalize_month_day(target_month_day)
    except ValueError:
        return False
    meta = (metadata_month_day or "").strip()
    if meta:
        try:
            if normalize_month_day(meta) == target:
                return True
        except ValueError:
            pass
        # Metadata says a different day — do not override with a loose substring.
        try:
            if normalize_month_day(meta) != target:
                extracted = extract_incident_month_days(text)
                return target in extracted
        except ValueError:
            pass

    extracted = extract_incident_month_days(text)
    return target in extracted


def infer_month_day_from_query(query: str) -> str:
    """Recover MM-DD from a search query so web hits can be date-guarded."""
    text = (query or "").strip()
    if not text:
        return ""
    token = re.search(r"(?<!\d)(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(?!\d)", text)
    if token:
        return f"{token.group(1)}-{token.group(2)}"
    chinese = _CHINESE_YMD_IN_TEXT.search(text)
    if chinese:
        return _pad_md(chinese.group("month"), chinese.group("day"))
    extracted = extract_incident_month_days(text)
    if len(extracted) == 1:
        return next(iter(extracted))
    return ""


def _web_item_blob(item: dict[str, str]) -> str:
    return f"{item.get('title') or ''} {item.get('content') or ''} {item.get('url') or ''}"


def filter_same_date_web_hits(results: list[dict[str, str]], month_day: str) -> list[dict[str, str]]:
    return [
        item
        for item in results
        if passes_same_date_guardrail(_web_item_blob(item), month_day)
    ]


def is_work_accident_related(text: str) -> bool:
    """True if the text looks like a workplace / industrial accident report."""
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in ACCIDENT_KEYWORDS)


def ensure_events_db() -> None:
    """Raise if the SQLite events DB has not been built yet."""
    if not db_ready(EVENTS_DB_PATH):
        raise KnowledgeBaseError(
            f"本地事件資料庫尚未建立（預期路徑：{EVENTS_DB_PATH}）。"
            "請先執行：python document_ingest.py"
        )


def get_serper() -> SmartSearchTool:
    """Shared Serper client for web / labour-department search."""
    global _serper
    if _serper is None:
        _serper = SmartSearchTool(default_max_results=8)
    return _serper


def _is_labour_accident_item(text: str) -> bool:
    """Keep fatal / workplace accident notices; drop heat warnings and job fairs."""
    if any(token in text for token in ("工作意外", "致命工作", "工業意外", "職安意外")):
        return True
    if "意外" in text and any(token in text for token in ("調查", "高度關注", "地盤", "工傷")):
        return True
    return False


def fetch_labour_listing_for_day(month_day: str) -> list[dict[str, str]]:
    """Read the Labour Department press-release listing and keep same MM-DD accident items."""
    month_day = normalize_month_day(month_day)
    try:
        request = urllib.request.Request(
            LABOUR_DEPT_NEWS_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SafetyAwarenessPulse/1.0)"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            html_text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return []

    plain = re.sub(r"<script[\s\S]*?</script>", " ", html_text, flags=re.I)
    plain = re.sub(r"<style[\s\S]*?</style>", " ", plain, flags=re.I)
    plain = re.sub(r"<[^>]+>", "\n", plain)
    plain = html.unescape(plain)
    lines = [re.sub(r"\s+", " ", line).strip() for line in plain.splitlines()]
    lines = [line for line in lines if line]

    date_re = re.compile(r"^(20\d{2})-(\d{2})-(\d{2})$")
    hits: list[dict[str, str]] = []
    for index, line in enumerate(lines):
        match = date_re.match(line)
        if not match:
            continue
        if f"{match.group(2)}-{match.group(3)}" != month_day:
            continue
        title = lines[index - 1] if index else ""
        blob = f"{title} {line}"
        if not _is_labour_accident_item(blob):
            continue
        iso_date = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        hits.append(
            {
                "title": title,
                "url": LABOUR_DEPT_NEWS_URL,
                "content": f"{title}（新聞公報日期：{iso_date}）",
            }
        )
    return hits


def labour_department_search(month_day: str, extra_query: str = "") -> dict[str, Any]:
    """Search only the Labour Department website, starting from the official listing URL."""
    month_day = normalize_month_day(month_day)
    query = " ".join(
        part
        for part in (
            extra_query.strip(),
            "香港勞工處 新聞公報 工作意外 工業意外 致命",
            " ".join(date_keyword_variants(month_day, with_years=False)),
            LABOUR_DEPT_NEWS_URL,
        )
        if part
    )

    listing_hits = fetch_labour_listing_for_day(month_day)
    serper_hits: list[dict[str, str]] = []
    serper_error = ""
    try:
        serper_hits = get_serper().search_sync(
            query,
            max_results=8,
            language="zh-hk",
            news=True,
            include_domains=list(LABOUR_DEPT_DOMAINS),
        )
        serper_hits = [
            item
            for item in serper_hits
            if "labour.gov.hk" in (item.get("url") or "").lower() or not item.get("url")
        ]
    except Exception as exc:  # noqa: BLE001
        serper_error = str(exc)

    merged: list[dict[str, str]] = list(listing_hits)
    seen = {(item.get("title"), item.get("url")) for item in merged}
    for item in serper_hits:
        key = (item.get("title"), item.get("url"))
        if key in seen:
            continue
        blob = _web_item_blob(item)
        if listing_hits and not _is_labour_accident_item(blob) and "labour.gov.hk" not in (item.get("url") or "").lower():
            continue
        if not passes_same_date_guardrail(blob, month_day):
            continue
        merged.append(item)
        seen.add(key)

    if merged:
        return {"status": "ok", "results": merged, "month_day": month_day}
    if serper_error:
        return {"status": "api_error", "error": serper_error, "results": [], "month_day": month_day}
    return {"status": "empty", "results": [], "month_day": month_day}


def _format_rag_hits(hits: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        meta = hit.get("metadata") or {}
        header = (
            f"[{index}] 來源檔案：{meta.get('source_file', '')} ｜ "
            f"投影片：{meta.get('slide_number', '')} ｜ "
            f"事故日期：{meta.get('iso_date') or '未知'} ｜ "
            f"月日：{meta.get('month_day') or '未知'}"
        )
        blocks.append(header + "\n" + hit.get("content", ""))
    return "\n\n-----\n\n".join(blocks)


def retrieve_local_work_accidents(month_day: str, extra_query: str = "") -> dict[str, Any]:
    """
    Look up SQLite events by month_day (MM-DD).

    Same calendar day may have multiple accident rows. For the daily alert,
    exactly one event is chosen at random when several match.
    """
    month_day = normalize_month_day(month_day)
    ensure_events_db()
    candidates = fetch_by_month_day(month_day, extra_query=extra_query)
    if not candidates:
        return {
            "month_day": month_day,
            "hits": [],
            "candidate_count": 0,
            "retrieval": "none",
        }

    event = random.choice(candidates)
    hit = {
        "content": event["content"],
        "metadata": {
            "id": event.get("id"),
            "iso_date": event["iso_date"],
            "month_day": event["month_day"],
            "title": event["title"],
            "location": event["location"],
            "category": event["category"],
            "source_file": event["source_file"],
            "slide_number": event["slide_number"],
            "language": event["language"],
        },
    }
    return {
        "month_day": month_day,
        "hits": [hit],
        "candidate_count": len(candidates),
        "retrieval": "sqlite_date_random",
    }


def web_search(query: str) -> dict[str, Any]:
    """Call Serper and normalize empty / error cases."""
    query = (query or "").strip()
    if not query:
        return {"status": "empty_query", "results": []}
    try:
        results = get_serper().search_sync(
            query,
            max_results=5,
            language="zh-hk",
            news=False,
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "api_error", "error": str(exc), "results": []}

    nonempty = [item for item in results if (item.get("content") or item.get("title"))]
    if not nonempty:
        return {"status": "empty", "results": []}
    return {"status": "ok", "results": nonempty}


def _format_web_hits(results: list[dict[str, str]]) -> str:
    blocks = []
    for index, item in enumerate(results, start=1):
        blocks.append(
            f"[{index}] {item.get('title') or '（無標題）'}\n"
            f"網址：{item.get('url') or '（無網址）'}\n"
            f"{item.get('content') or ''}"
        )
    return "\n\n".join(blocks)


@tool
def search_local_work_accidents(month_day: str, extra_query: str = "") -> str:
    """Search the local SQLite events DB (built from the 4 PPTs) for working /
    industrial accidents on the same month-day (MM-DD) in previous years.

    Lookup is by date: WHERE month_day = MM-DD. Multiple accidents on the same
    day are stored as separate rows; this tool returns exactly one of them
    (random pick) for the daily alert.

    ALWAYS call this tool first when generating the daily safety alert.
    Any workplace accident is accepted (construction, factory, railway, etc.).
    Keep extra_query in Traditional Chinese when possible. Do not translate PPT text.

    Args:
        month_day: Target month and day, preferably MM-DD (example: 04-04).
        extra_query: Optional extra keywords such as 工業意外 or a location name.
    """
    try:
        payload = retrieve_local_work_accidents(month_day, extra_query=extra_query)
    except KnowledgeBaseError as exc:
        return f"[ERROR] {exc}"
    except ValueError as exc:
        return f"[ERROR] 日期格式無效：{exc}"
    except Exception as exc:  # noqa: BLE001
        return f"[ERROR] 本地知識庫檢索失敗：{exc}"

    hits = payload["hits"]
    if not hits:
        return (
            f"[NO_HITS] 本地 PPT 事件庫沒有找到 {payload['month_day']} "
            "的工作意外紀錄。請改用 search_labour_department。"
        )
    candidate_count = int(payload.get("candidate_count") or len(hits))
    if candidate_count > 1:
        header = (
            f"[HITS] 本地事件庫在 {payload['month_day']} 找到 {candidate_count} 筆工作意外，"
            "今日警示隨機抽取其中 1 筆。以下為原文（繁體中文，未經翻譯）：\n\n"
        )
    else:
        header = (
            f"[HITS] 本地事件庫找到 1 筆 {payload['month_day']} 的工作意外紀錄。"
            "以下為原文（繁體中文，未經翻譯）：\n\n"
        )
    return header + _format_rag_hits(hits)


@tool
def search_labour_department(month_day: str, extra_query: str = "") -> str:
    """Search Hong Kong Labour Department press releases first, especially
    https://www.labour.gov.hk/tc/major/content.php

    ALWAYS call this as layer 2 after search_local_work_accidents returns NO_HITS.
    Do not use the open web until this tool also returns NO_HITS.

    Args:
        month_day: Target month and day, preferably MM-DD (example: 09-11).
        extra_query: Optional keywords such as 工作意外 or 致命.
    """
    try:
        payload = labour_department_search(month_day, extra_query=extra_query)
    except ValueError as exc:
        return f"[ERROR] 日期格式無效：{exc}"
    except Exception as exc:  # noqa: BLE001
        return f"[API_ERROR] 勞工處新聞公報搜尋失敗：{exc}"

    if payload["status"] == "api_error":
        return f"[API_ERROR] 勞工處網站／Serper 搜尋失敗：{payload.get('error', 'unknown error')}"
    if payload["status"] == "empty" or not payload.get("results"):
        return (
            f"[NO_HITS] 勞工處新聞公報（{LABOUR_DEPT_NEWS_URL}）沒有找到 "
            f"{payload.get('month_day', month_day)} 的工作意外。請改用 search_web。"
        )
    header = (
        f"[HITS] 勞工處新聞公報找到 {len(payload['results'])} 筆 "
        f"{payload.get('month_day', month_day)} 相關紀錄。"
        f"來源優先：{LABOUR_DEPT_NEWS_URL}\n\n"
    )
    return header + _format_web_hits(payload["results"])


@tool
def search_web(query: str, month_day: str = "") -> str:
    """Search the live web with Serper. Use this only after BOTH local RAG and
    search_labour_department returned NO_HITS (or for follow-up questions).

    Write the target date in several forms in the query (9月16日, 09-16, 9.16,
    16 September). Hits that are not that same MM-DD are dropped.

    Suggested queries by fallback level:
      Level 3: world working / industrial / workplace accident on this day in history.
      Level 4: interesting historical facts / 歷史上的今天 on this day.

    Args:
        query: Search query. Include the month-day written in several forms.
        month_day: Optional MM-DD guardrail. Inferred from query when omitted.
    """
    inferred = ""
    if (month_day or "").strip():
        try:
            inferred = normalize_month_day(month_day)
        except ValueError:
            inferred = infer_month_day_from_query(query)
    else:
        inferred = infer_month_day_from_query(query)

    expanded = query
    if inferred:
        extra = " ".join(date_keyword_variants(inferred, with_years=False))
        if extra not in query:
            expanded = f"{query} {extra}".strip()

    try:
        payload = web_search(expanded)
    except Exception as exc:  # noqa: BLE001
        return f"[API_ERROR] 網路搜尋連線失敗：{exc}"

    if payload["status"] == "empty_query":
        return "[EMPTY_SEARCH] 搜尋字串為空，請提供包含日期與主題的查詢。"
    if payload["status"] == "api_error":
        return f"[API_ERROR] Serper 搜尋失敗：{payload.get('error', 'unknown error')}"
    if payload["status"] == "empty":
        return f"[EMPTY_SEARCH] 網路上沒有找到可用結果。查詢：{expanded}"

    results = payload["results"]
    if inferred:
        results = filter_same_date_web_hits(results, inferred)
        if not results:
            return (
                f"[EMPTY_SEARCH] 搜尋結果均不是 {inferred} 當日的紀錄，已全部剔除。"
                f"查詢：{expanded}"
            )
    return "[WEB_HITS] 網路搜尋結果：\n\n" + _format_web_hits(results)


SYSTEM_PROMPT = """You are the Daily Safety Alert assistant for workplace safety staff.

Tool / fallback rules (do not describe these rules in the user-visible answer):
1. Do not invent accidents. If a layer has no usable hit, go to the next layer.
2. Daily alert order, stop at the first usable hit:
   ① Call search_local_work_accidents (local SQLite events from the 4 PPTs) for any workplace accident on that MM-DD.
      If [HITS] → write the safety notice, then stop.
   ② If [NO_HITS] or [ERROR] → call search_labour_department for the same MM-DD.
      This tool searches the Hong Kong Labour Department press-release list first:
      https://www.labour.gov.hk/tc/major/content.php
      If [HITS] → write the notice from that official source, then stop.
   ③ If Labour Department also returns [NO_HITS] → call search_web for worldwide workplace / industrial accidents on that day.
      If found → write the notice, then stop.
   ④ If no workplace accident of any kind is found → call search_web for "on this day in history" and output that fact.
3. Date guardrail: only use an incident whose event date is the requested MM-DD.
   Reject nearby dates (9.17 when asking 9.16) even if the text contains 16 or 17 as a count, age, or year fragment.
   When calling search_web, pass month_day and write the date in several forms (9月16日, 09-16, 9.16, 16 September).
4. Notice structure: title with layer, date and place, summary, 2–4 practical safety reminders, sources.
5. Never translate or rewrite the stored PPT text in SQLite. When quoting a PPT, keep the original Traditional Chinese wording. In English output mode, quote the original then add an English paraphrase.
6. Follow the OUTPUT_LANGUAGE tag in the latest user message:
   - zh-Hant: every user-visible sentence must be Traditional Chinese (Hong Kong wording). No English sentences. No English process notes. Proper nouns such as ICU may stay as-is.
   - en: every user-visible sentence must be English, except original PPT quotes.
7. The final answer must NEVER contain process / debug narration, including:
   "Level ①/②/③/④ returned...", "I'll output the historical fact", "Since no workplace accident was found",
   "依規則在此停止", tool names, [HITS], [NO_HITS], [WEB_HITS], [EMPTY_SEARCH].
   Those belong only in internal tool use, not in the text shown to the user.
"""

LEVEL_LABELS = {
    "local": {"zh-Hant": "① 本地 PPT 工作意外", "en": "① Local PPT workplace accident"},
    "labour_dept": {"zh-Hant": "② 勞工處新聞公報", "en": "② Labour Department press release"},
    "hk_web": {"zh-Hant": "② 勞工處新聞公報", "en": "② Labour Department press release"},
    "world_web": {"zh-Hant": "③ 全球工作意外", "en": "③ Worldwide workplace accident"},
    "history": {"zh-Hant": "④ 歷史冷知識", "en": "④ Historical fact"},
    "unknown": {"zh-Hant": "安全警示", "en": "Safety alert"},
}

_PROCESS_LINE_RE = re.compile(
    r"(?im)^(?:\s*(?:level\s*[①②③④1-4]|layer\s*[1-4]).*(?:returned|usable|historical fact).*|"
    r".*since no workplace accident was found.*|"
    r".*i(?:'|’)?ll output the historical fact.*|"
    r".*no workplace accident was found at any level.*|"
    r".*依規則在此停止.*|"
    r".*不再搜尋網絡.*|"
    r".*本地知識庫已有.*依規則.*)\s*$"
)


def normalize_output_language(language: str | None) -> str:
    text = (language or "zh-Hant").strip().lower()
    if text in {"en", "en-us", "en-gb", "english"}:
        return "en"
    return "zh-Hant"


def language_instruction(language: str) -> str:
    """Prefix that forces the model to write only in the requested UI language."""
    language = normalize_output_language(language)
    if language == "en":
        return (
            "OUTPUT_LANGUAGE=en\n"
            "Write the entire user-visible answer in English. "
            "You may quote original Traditional Chinese PPT wording, then paraphrase it in English. "
            "Do not describe tools, layers, or search steps."
        )
    return (
        "OUTPUT_LANGUAGE=zh-Hant\n"
        "給使用者看的全部文字必須是繁體中文，不可夾雜英文句子或英文過程說明。"
        "禁止寫出 Level ④ returned...、I'll output... 這類後台語句。"
        "不要提及工具名稱或搜尋過程。"
    )


def sanitize_user_facing_text(text: str) -> str:
    """Drop leaked English/Chinese process narration from the model output."""
    if not text:
        return text
    kept = [line for line in text.splitlines() if not _PROCESS_LINE_RE.match(line)]
    cleaned = "\n".join(kept).strip()
    return cleaned or text.strip()


def build_llm() -> AzureChatOpenAI:
    """Azure OpenAI chat model (gpt-5.4-mini) with tool / function calling."""
    api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
    if not api_key or api_key.startswith("<") or api_key == "your-api-key":
        raise RuntimeError(
            "Missing AZURE_OPENAI_API_KEY. Copy .env.example to .env and fill in the key."
        )
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or AZURE_OPENAI_ENDPOINT).rstrip("/") + "/"
    deployment = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or AZURE_OPENAI_DEPLOYMENT).strip()
    api_version = (os.getenv("AZURE_OPENAI_API_VERSION") or AZURE_OPENAI_API_VERSION).strip()
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=deployment,
        max_retries=2,
        timeout=120,
        # GPT-5 family uses max_completion_tokens; temperature is left at API default.
        model_kwargs={"max_completion_tokens": 16384},
    )


def build_agent():
    """LangChain tool-calling agent powered by Azure OpenAI function calling."""
    global _agent
    if _agent is not None:
        return _agent

    llm = build_llm()
    tools = [search_local_work_accidents, search_labour_department, search_web]
    try:
        _agent = _create_agent(
            model=llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )
    except TypeError:
        # langgraph.prebuilt.create_react_agent uses prompt= instead of system_prompt=.
        _agent = _create_agent(
            model=llm,
            tools=tools,
            prompt=SYSTEM_PROMPT,
        )
    return _agent


def daily_alert_user_prompt(target: date, language: str = "zh-Hant") -> str:
    month_day = to_month_day(target)
    month = target.month
    day = target.day
    language = normalize_output_language(language)
    closing = (
        "Write only the user-visible English safety notice or historical fact. No process narration."
        if language == "en"
        else "最後只輸出給使用者看的繁體中文警示或史實，不要寫任何英文過程說明。"
    )
    return (
        f"{language_instruction(language)}\n\n"
        f"請產生「每日安全警示」。香港今日日期是 {target.isoformat()}，月日為 {month_day}。\n"
        "請嚴格執行四層備援，完成其中一層後立刻停止：\n"
        f"① 先呼叫 search_local_work_accidents，month_day={month_day}。\n"
        f"② 若 [NO_HITS]，呼叫 search_labour_department，month_day={month_day}。"
        f"必須先查勞工處新聞公報 {LABOUR_DEPT_NEWS_URL}。\n"
        f"③ 若勞工處也是 [NO_HITS]，呼叫 search_web，month_day={month_day}，查："
        f"world workplace industrial working accident on {month_day} in history "
        f"{month}月{day}日 {month}.{day} {day} {_EN_MONTHS[month]} 工業意外 工作意外 工傷。"
        f"只用 {month_day} 當日的意外。\n"
        f"④ 若完全沒有工作意外，呼叫 search_web，month_day={month_day}，查："
        f"historical events on {month_day} interesting facts 歷史上的今天 "
        f"{month}月{day}日 {month}.{day}。\n"
        f"{closing}"
    )


def wrap_user_message(user_text: str, language: str = "zh-Hant") -> str:
    """Keep follow-up questions in the same output language as the UI."""
    return f"{language_instruction(language)}\n\n{user_text}"


def extract_final_text(result: dict[str, Any], language: str = "zh-Hant") -> str:
    language = normalize_output_language(language)
    empty = "No message was returned." if language == "en" else "系統沒有回傳任何訊息。"
    fallback = "The system did not produce a usable reply." if language == "en" else "系統沒有產生可用的回覆。"
    messages = result.get("messages") or []
    if not messages:
        return empty
    last = messages[-1]
    content = getattr(last, "content", None)
    if isinstance(content, str) and content.strip():
        return sanitize_user_facing_text(content)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        joined = "\n".join(part for part in parts if part).strip()
        if joined:
            return sanitize_user_facing_text(joined)
    return sanitize_user_facing_text(str(content or fallback))


def summarize_tool_trace(result: dict[str, Any]) -> list[str]:
    """Collect which tools the agent actually called (for the demo UI / testing)."""
    trace: list[str] = []
    for message in result.get("messages") or []:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            trace.append(f"{name}({args})")
    return trace


def invoke_agent(
    user_text: str,
    history: list[dict[str, str]] | None = None,
    recursion_limit: int = 16,
    language: str = "zh-Hant",
) -> dict[str, Any]:
    """Run the Azure OpenAI tool-calling agent and return text + tool trace."""
    language = normalize_output_language(language)
    agent = build_agent()
    messages: list[dict[str, str]] = []
    for item in history or []:
        role = item.get("role") or "user"
        content = item.get("content") or ""
        if role in {"user", "assistant", "system"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})

    try:
        result = agent.invoke({"messages": messages}, config={"recursion_limit": recursion_limit})
    except Exception as exc:  # noqa: BLE001
        fail = (
            f"Sorry, the model call failed: {exc}"
            if language == "en"
            else f"抱歉，連線或模型呼叫失敗：{exc}"
        )
        return {
            "text": fail,
            "trace": [],
            "error": str(exc),
        }

    return {
        "text": extract_final_text(result, language=language),
        "trace": summarize_tool_trace(result),
        "error": None,
        "raw": result,
    }


def generate_daily_alert(target: date | None = None, language: str = "zh-Hant") -> dict[str, Any]:
    """Ask the agent to produce today's safety alert using the 4-level fallback."""
    language = normalize_output_language(language)
    target = target or today_in_hong_kong()
    month_day = to_month_day(target)
    outcome = invoke_agent(daily_alert_user_prompt(target, language=language), language=language)
    level_key = infer_fallback_level(outcome.get("trace") or [], outcome.get("text") or "")
    outcome["month_day"] = month_day
    outcome["iso_date"] = target.isoformat()
    outcome["level_key"] = level_key
    outcome["level_guess"] = LEVEL_LABELS.get(level_key, LEVEL_LABELS["unknown"])[language]
    outcome["language"] = language
    return outcome


def _count_zh_chars(text: str) -> int:
    """Count characters for the Telegram short-reminder length check (ignore whitespace)."""
    return len(re.sub(r"\s+", "", text or ""))


def _clip_reminder(text: str, *, min_chars: int = 30, max_chars: int = 50) -> str:
    """Keep Traditional Chinese reminder roughly within 30–50 characters."""
    cleaned = re.sub(r"\s+", "", (text or "").strip())
    cleaned = cleaned.strip("「」『』\"'。！？!?；;，,、")
    if not cleaned:
        return "今日請留意職安風險，工作前先檢查防護。"
    if _count_zh_chars(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    if _count_zh_chars(cleaned) < min_chars:
        pad = "請提高警覺，遵守安全守則。"
        if cleaned and cleaned[-1] not in "。！？!?":
            cleaned = cleaned + "。"
        cleaned = (cleaned + pad)[:max_chars]
    return cleaned


def _fallback_short_from_evidence(evidence: str) -> str:
    """Heuristic short line when the LLM is unavailable."""
    lines = [line.strip() for line in (evidence or "").splitlines() if line.strip()]
    body = ""
    for line in lines:
        if line.startswith("[") or line.startswith("來源") or line.startswith("網址"):
            continue
        if "-----" in line:
            continue
        body = line
        break
    if not body and lines:
        body = lines[-1]
    # Prefer a practical reminder framing.
    if "意外" in body or "事故" in body:
        return _clip_reminder(f"往年今日曾發生工業意外，請提高警覺並做好防護。")
    return _clip_reminder(body or "今日請留意職安風險，工作前先檢查防護。")


def generate_short_reminder(
    target: date | None = None,
) -> dict[str, Any]:
    """
    Build a Discord-ready safety reminder: Traditional Chinese, ~30–50 characters.

    Uses the deterministic 4-level pipeline for evidence, then Azure OpenAI to compress.
    Returns detail_text (full event evidence) for the in-Discord "了解更多" button.
    """
    target = target or today_in_hong_kong()
    month_day = to_month_day(target)
    iso_date = target.isoformat()

    pipeline = run_priority_pipeline(month_day)
    evidence = str(pipeline.get("evidence") or "")[:3500]
    level_label = str(pipeline.get("label") or "")
    detail_text = _format_detail_for_discord(
        iso_date=iso_date,
        month_day=month_day,
        level_label=level_label,
        evidence=evidence,
    )

    prompt = (
        "你是職場安全短訊編輯。根據下列資料，寫一句繁體中文（香港用詞）每日安全提醒。\n"
        "硬性要求：\n"
        "1. 只輸出提醒正文本身，不要標題、不要引號、不要來源、不要連結。\n"
        "2. 長度必須約 30 至 50 個字（含標點），寧可精簡。\n"
        "3. 內容要可執行（提醒防護／檢查／程序），不要長篇敘述案情。\n"
        f"日期：{iso_date}（月日 {month_day}）\n"
        f"資料層級：{level_label}\n"
        f"資料：\n{evidence}\n"
    )

    text = ""
    error = None
    try:
        llm = build_llm()
        response = llm.invoke(prompt)
        text = getattr(response, "content", None) or str(response)
        if isinstance(text, list):
            text = "".join(str(part) for part in text)
        text = _clip_reminder(str(text).strip())
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        text = _fallback_short_from_evidence(evidence)

    return {
        "text": text,
        "iso_date": iso_date,
        "month_day": month_day,
        "detail_text": detail_text,
        "level_label": level_label,
        "char_count": _count_zh_chars(text),
        "error": error,
    }


def _format_detail_for_discord(
    *,
    iso_date: str,
    month_day: str,
    level_label: str,
    evidence: str,
) -> str:
    """Format the full event evidence shown when the user clicks 了解更多."""
    body = (evidence or "").strip() or "（暫無詳細資料）"
    header = (
        f"【事件詳情】\n"
        f"日期：{iso_date}（月日 {month_day}）\n"
        f"來源層級：{level_label or '未知'}\n"
        f"{'─' * 16}\n"
    )
    # Discord message limit is 2000; leave room for header when splitting later.
    return header + body


def infer_fallback_level(trace: list[str], text: str) -> str:
    """Best-effort key for which fallback level produced the answer."""
    rag_called = any("search_local_work_accidents" in step for step in trace)
    labour_called = any("search_labour_department" in step for step in trace)
    web_calls = [step for step in trace if step.startswith("search_web")]
    if rag_called and not labour_called and not web_calls:
        return "local"
    joined = " ".join(web_calls).lower() + " " + text
    if any(token in joined for token in ("歷史上的今天", "historical events", "interesting facts", "冷知識")):
        return "history"
    if any(token in joined for token in ("world workplace", "industrial working accident", "全球工作意外", "全球工業意外")):
        return "world_web"
    if labour_called and not web_calls:
        return "labour_dept"
    if web_calls:
        return "world_web"
    if labour_called:
        return "labour_dept"
    if rag_called:
        return "local"
    return "unknown"


def run_priority_pipeline(month_day: str) -> dict[str, Any]:
    """
    Deterministic 4-level fallback used by the testing guide.

    This does not rely on the LLM choosing tools; it calls the same two tools in order.
    """
    month_day = normalize_month_day(month_day)

    level1 = search_local_work_accidents.invoke({"month_day": month_day, "extra_query": "工業意外"})
    if level1.startswith("[HITS]"):
        return {"level": 1, "label": "① 本地 PPT 工作意外", "evidence": level1}

    level2 = search_labour_department.invoke({"month_day": month_day, "extra_query": "工作意外"})
    if level2.startswith("[HITS]"):
        return {"level": 2, "label": "② 勞工處新聞公報", "evidence": level2}

    world_query = (
        f"world workplace industrial working accident on {month_day} in history "
        f"{' '.join(date_keyword_variants(month_day, with_years=False))} 工業意外 工作意外 工傷"
    )
    level3 = search_web.invoke({"query": world_query, "month_day": month_day})
    if level3.startswith("[WEB_HITS]") and is_work_accident_related(level3):
        return {"level": 3, "label": "③ 全球工作意外", "evidence": level3}

    fact_query = (
        f"historical events on {month_day} interesting facts 歷史上的今天 "
        f"{' '.join(date_keyword_variants(month_day, with_years=False))}"
    )
    level4 = search_web.invoke({"query": fact_query, "month_day": month_day})
    return {"level": 4, "label": "④ 歷史冷知識", "evidence": level4}


def interactive_chat(start_date: date, language: str = "zh-Hant") -> None:
    language = normalize_output_language(language)
    print("=" * 60)
    if language == "en":
        print("Daily safety alert chatbot (type exit to quit)")
        print("=" * 60)
        print("Generating today's alert, please wait...\n")
    else:
        print("每日安全警示聊天機械人（輸入 exit 結束）")
        print("=" * 60)
        print("正在產生今日警示，請稍候...\n")
    alert = generate_daily_alert(start_date, language=language)
    print(alert["text"])
    if language == "en":
        print("\nTools:", alert["trace"] or "(none)")
        print("Level:", alert["level_guess"])
        print("\nWant to know more? Ask here.\n")
        prompt_label = "You: "
        assistant_label = "Assistant:"
        bye = "Goodbye."
    else:
        print("\n工具呼叫：", alert["trace"] or "（無）")
        print("判定層級：", alert["level_guess"])
        print("\n想了解更多？在此提問\n")
        prompt_label = "你："
        assistant_label = "助手："
        bye = "再見。"

    history = [
        {"role": "user", "content": daily_alert_user_prompt(start_date, language=language)},
        {"role": "assistant", "content": alert["text"]},
    ]
    while True:
        try:
            user_input = input(prompt_label).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{bye}")
            break
        if user_input.lower() in {"exit", "quit", "q"}:
            print(bye)
            break
        if not user_input:
            continue
        reply = invoke_agent(
            wrap_user_message(user_input, language),
            history=history,
            language=language,
        )
        print(f"\n{assistant_label}", reply["text"], "\n")
        history.append({"role": "user", "content": wrap_user_message(user_input, language)})
        history.append({"role": "assistant", "content": reply["text"]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Workplace safety-alert agent (Azure OpenAI + SQLite + Serper).")
    parser.add_argument("--date", help="Override Hong Kong date, YYYY-MM-DD or MM-DD.")
    parser.add_argument("--chat", action="store_true", help="Start an interactive terminal chatbot.")
    parser.add_argument(
        "--lang",
        choices=["zh-Hant", "en"],
        default="zh-Hant",
        help="User-facing output language. Does not change SQLite retrieval.",
    )
    parser.add_argument(
        "--test-fallback",
        action="store_true",
        help="Run the deterministic 4-level pipeline and print which level fired.",
    )
    args = parser.parse_args()
    target = today_in_hong_kong(args.date)

    if args.test_fallback:
        print(f"[test] Target date: {target.isoformat()}  MM-DD={to_month_day(target)}")
        result = run_priority_pipeline(to_month_day(target))
        print(f"[test] Fired level: {result['label']}")
        print(result["evidence"][:4000])
        return

    if args.chat:
        interactive_chat(target, language=args.lang)
        return

    alert = generate_daily_alert(target, language=args.lang)
    print(alert["text"])
    print("\n---")
    print("MM-DD:", alert["month_day"])
    print("Tools:", alert["trace"] or ("(none)" if args.lang == "en" else "（無）"))
    print("Level:", alert["level_guess"])


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Testing guide: verifying the 4-level fallback logic
# ---------------------------------------------------------------------------
# Prerequisites:
#   1) Copy .env.example to .env and set AZURE_OPENAI_API_KEY + SERPER_API_KEY
#   2) python document_ingest.py          # build data/events.db from ./assets/ppts
#   3) python agent_mtr_bot.py --chat     # or: streamlit run streamlit_app.py
#
# The PPTs are industrial-accident newspaper cuttings (not limited to MTR).
# Example records (DD-MM-YYYY):
#   - 29-12-2023  元朗工業邨燒焊金屬鐡擊中工人
#   - 04-04-2023  港鐵灣仔站維修工於路軌跌倒
#   - 01-03-2023  港鐵旺角站扶手梯維修
#
# Level ① Local SQLite date lookup (must stop here, do not call Serper):
#   python agent_mtr_bot.py --date 2026-12-29
#   python agent_mtr_bot.py --date 2026-04-04
#   Expected: tool trace contains only search_local_work_accidents
#             and the notice quotes original Traditional Chinese PPT wording.
#
# Level ② Hong Kong working-accident web search:
#   Pick an MM-DD that has NO accident slide in the 4 PPTs.
#   python agent_mtr_bot.py --date YYYY-MM-DD --test-fallback
#   Expected: Level 1 returns [NO_HITS], then search_web looks for Hong Kong
#             workplace / industrial accidents on that day in history.
#
# Level ③ World working-accident web search:
#   If level ② also returns [EMPTY_SEARCH] (or no HK workplace hit), the next
#   search_web call must be a global workplace / industrial accident query
#   for the same MM-DD. Watch the printed evidence / Streamlit tool trace.
#
# Level ④ Historical fact:
#   If levels 1–3 all miss, the last search_web query must be "historical events" /
#   「歷史上的今天」. The user-facing output is still Traditional Chinese, but it is
#   a fact rather than a workplace safety notice.
#
# Empty / error handling checks:
#   - Delete or rename data/events.db then call local search → [ERROR] asking you to ingest first.
#   - Temporarily set a bad SERPER_API_KEY → [API_ERROR] without crashing the app.
#   - Agent follow-up: after an alert is shown, ask「這次意外的主要風險是甚麼？」
#     The chatbot should answer in Traditional Chinese, optionally calling tools again.
#
# Streamlit demo:
#   streamlit run streamlit_app.py
#   Use the sidebar date picker to force a PPT date (level ①) vs a date with
#   no local cutting (level ②–④).
# ---------------------------------------------------------------------------
