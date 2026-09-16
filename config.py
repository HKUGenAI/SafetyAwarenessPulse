"""Shared paths and runtime constants for the workplace safety RAG agent."""

from pathlib import Path

# Project root (this file lives at the repository root).
PROJECT_ROOT = Path(__file__).resolve().parent

# Source PPTs written in Traditional Chinese.
PPT_DIR = PROJECT_ROOT / "assets" / "ppts"

# Local ChromaDB persistence folder (not cloud).
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "mtr_safety_ppt"

# DeepSeek chat model (function calling is supported by deepseek-chat).
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Local multilingual embedding model. DeepSeek has no public embedding API,
# so we embed Traditional Chinese text locally and keep original wording.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Hong Kong local date is used for "today" (MM-DD matching).
HONG_KONG_TZ = "Asia/Hong_Kong"

# Retrieval / chunk settings.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RETRIEVAL_K = 8

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
