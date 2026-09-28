"""Shared paths and runtime constants for the workplace safety agent."""

from __future__ import annotations

import os
from pathlib import Path

# Project root (this file lives at the repository root).
PROJECT_ROOT = Path(__file__).resolve().parent

# Source PPTs written in Traditional Chinese.
PPT_DIR = PROJECT_ROOT / "assets" / "ppts"

# Local SQLite store of structured events.
DATA_DIR = PROJECT_ROOT / "data"
EVENTS_DB_PATH = DATA_DIR / "events.db"

# DeepSeek chat model (function calling is supported by deepseek-chat).
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Official Hong Kong Labour Department press-release listing (layer 2 after local DB).
LABOUR_DEPT_NEWS_URL = "https://www.labour.gov.hk/tc/major/content.php"
LABOUR_DEPT_DOMAINS = ("labour.gov.hk", "www.labour.gov.hk")

# Hong Kong local date is used for "today" (MM-DD matching).
HONG_KONG_TZ = "Asia/Hong_Kong"

# Discord channel broadcast (short reminder + in-app event details).
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID", "").strip()
DISCORD_BROADCAST_HOUR = int(os.getenv("DISCORD_BROADCAST_HOUR", "9"))
DISCORD_BROADCAST_MINUTE = int(os.getenv("DISCORD_BROADCAST_MINUTE", "0"))

# Keywords used to decide whether a web result is a workplace / industrial accident.
# Original Traditional Chinese terms are kept as-is (no translation of PPT text).
ACCIDENT_KEYWORDS = (
    "意外",
    "事故",
    "工傷",
    "工業意外",
    "工作意外",
    "職安",
    "職業安全",
    "地盤",
    "工場",
    "施工",
    "勞工處",
    "workplace accident",
    "industrial accident",
    "work accident",
    "occupational accident",
    "construction accident",
    "fatal accident",
)
