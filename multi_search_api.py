"""
Serper-backed SmartSearchTool.

Provides web / news search with optional recency filters, in-memory caching,
and light rate-limit tracking. Usable standalone, with LangChain agents, or
as a CrewAI-style tool (name / description / _run).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_NEWS_URL = "https://google.serper.dev/news"
PROVIDER_NAME = "serper"

# Soft client-side budget (Serper free tier is typically ~2500 queries / month).
DEFAULT_DAILY_LIMIT = 2500
DEFAULT_CACHE_TTL_SECONDS = 3600


def _days_to_tbs(days_back: int) -> str:
    """Map days_back to a Google tbs time filter."""
    days = max(1, int(days_back))
    if days <= 1:
        return "qdr:d"
    if days <= 7:
        return f"qdr:d{days}" if days < 7 else "qdr:w"
    if days <= 14:
        return f"qdr:d{days}"
    if days <= 31:
        return "qdr:m"
    if days <= 365:
        return "qdr:y"
    # Custom calendar range as a last resort.
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    return f"cdr:1,cd_min:{start.month}/{start.day}/{start.year},cd_max:{end.month}/{end.day}/{end.year}"


def _normalize_language(language: str | None) -> tuple[str, str]:
    """Return (hl, gl) for Serper from a language / locale hint."""
    raw = (language or "en").strip().lower().replace("_", "-")
    mapping = {
        "en": ("en", "us"),
        "en-us": ("en", "us"),
        "en-gb": ("en", "uk"),
        "zh": ("zh-tw", "hk"),
        "zh-tw": ("zh-tw", "hk"),
        "zh-hant": ("zh-tw", "hk"),
        "zh-hk": ("zh-tw", "hk"),
        "zh-cn": ("zh-cn", "cn"),
        "yue": ("zh-tw", "hk"),
    }
    return mapping.get(raw, (raw.split("-")[0] or "en", "us"))


class SmartSearchTool:
    """Serper Google Search wrapper with cache + rate-limit helpers."""

    name: str = "smart_search"
    description: str = (
        "Search the live web via Serper (Google). "
        "Use for recent news or general web lookup. "
        "Input should be a clear search query string."
    )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        daily_limit: int = DEFAULT_DAILY_LIMIT,
        default_max_results: int = 10,
    ) -> None:
        self.api_key = (api_key or os.getenv("SERPER_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError(
                "Missing SERPER_API_KEY. Copy .env.example to .env and fill in the key."
            )
        self.cache_ttl_seconds = cache_ttl_seconds
        self.daily_limit = daily_limit
        self.default_max_results = default_max_results
        self._cache_enabled = True
        self._cache: dict[str, dict[str, Any]] = {}
        self._rate: dict[str, Any] = {
            "day": datetime.now(timezone.utc).date().isoformat(),
            "count": 0,
            "limited": False,
            "last_error": "",
        }

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        language: str = "en",
        news: bool = False,
        days_back: int | None = None,
        include_domains: list[str] | None = None,
    ) -> list[dict[str, str]]:
        return await asyncio.to_thread(
            self.search_sync,
            query,
            max_results=max_results,
            language=language,
            news=news,
            days_back=days_back,
            include_domains=include_domains,
        )

    async def search_recent_content(
        self,
        query: str,
        max_results: int = 10,
        days_back: int = 14,
        language: str = "en",
    ) -> list[dict[str, str]]:
        """Search for content from the last `days_back` days (news-biased)."""
        return await self.search(
            query,
            max_results=max_results,
            language=language,
            news=True,
            days_back=days_back,
        )

    def search_sync(
        self,
        query: str,
        *,
        max_results: int | None = None,
        language: str = "en",
        news: bool = False,
        days_back: int | None = None,
        include_domains: list[str] | None = None,
    ) -> list[dict[str, str]]:
        query = (query or "").strip()
        if not query:
            return []

        num = max(1, min(int(max_results or self.default_max_results), 100))
        hl, gl = _normalize_language(language)
        effective_query = query
        if include_domains:
            sites = " OR ".join(f"site:{d}" for d in include_domains if d)
            if sites:
                effective_query = f"({sites}) {query}"

        cache_key = self._cache_key(
            effective_query, num=num, hl=hl, gl=gl, news=news, days_back=days_back
        )
        if self._cache_enabled:
            cached = self._get_cache(cache_key)
            if cached is not None:
                return cached

        self._roll_rate_day()
        if self._rate["limited"] or self._rate["count"] >= self.daily_limit:
            self._rate["limited"] = True
            raise RuntimeError(
                f"Serper daily rate limit reached ({self.daily_limit}). "
                "Call reset_rate_limits() or wait for the next UTC day."
            )

        payload: dict[str, Any] = {
            "q": effective_query,
            "num": num,
            "hl": hl,
            "gl": gl,
        }
        if days_back is not None:
            payload["tbs"] = _days_to_tbs(days_back)

        url = SERPER_NEWS_URL if news else SERPER_SEARCH_URL
        raw = self._post_json(url, payload)
        self._rate["count"] += 1

        results = self._parse_serper_payload(raw, news=news)
        if self._cache_enabled:
            self._set_cache(cache_key, results)
        return results

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        self._roll_rate_day()
        self.clear_cache(expired_only=True)
        return {
            "providers": [PROVIDER_NAME],
            "rate_limited_providers": [PROVIDER_NAME] if self._rate["limited"] else [],
            "rate_limits": {
                PROVIDER_NAME: {
                    "day": self._rate["day"],
                    "count": self._rate["count"],
                    "daily_limit": self.daily_limit,
                    "limited": self._rate["limited"],
                    "last_error": self._rate["last_error"],
                }
            },
            "cache": {
                "enabled": self._cache_enabled,
                "total_entries": len(self._cache),
                "ttl_seconds": self.cache_ttl_seconds,
            },
        }

    def clear_cache(self, expired_only: bool = False) -> int:
        """Clear cache. By default removes expired entries only (SmartSearchTool docs)."""
        if not expired_only:
            removed = len(self._cache)
            self._cache.clear()
            return removed
        now = time.time()
        expired = [key for key, entry in self._cache.items() if entry["expires_at"] <= now]
        for key in expired:
            del self._cache[key]
        return len(expired)

    def disable_cache(self) -> None:
        self._cache_enabled = False

    def enable_cache(self) -> None:
        self._cache_enabled = True

    # ------------------------------------------------------------------
    # Rate limit management
    # ------------------------------------------------------------------

    def reset_rate_limits(self) -> None:
        """Reset rate-limit tracking (e.g. new day / new quota)."""
        self._rate = {
            "day": datetime.now(timezone.utc).date().isoformat(),
            "count": 0,
            "limited": False,
            "last_error": "",
        }

    # ------------------------------------------------------------------
    # CrewAI / LangChain-style tool interface
    # ------------------------------------------------------------------

    def _run(self, query: str) -> str:
        results = self.search_sync(query)
        if not results:
            return "No search results found."
        blocks = []
        for index, item in enumerate(results, start=1):
            blocks.append(
                f"[{index}] {item.get('title') or '(no title)'}\n"
                f"URL: {item.get('url') or ''}\n"
                f"{item.get('content') or ''}"
            )
        return "\n\n".join(blocks)

    async def _arun(self, query: str) -> str:
        return await asyncio.to_thread(self._run, query)

    def run(self, query: str) -> str:
        return self._run(query)

    def __call__(self, query: str) -> str:
        return self._run(query)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _roll_rate_day(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if self._rate["day"] != today:
            self.reset_rate_limits()

    def _cache_key(self, query: str, **kwargs: Any) -> str:
        blob = json.dumps({"q": query, **kwargs}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _get_cache(self, key: str) -> list[dict[str, str]] | None:
        entry = self._cache.get(key)
        if not entry:
            return None
        if entry["expires_at"] <= time.time():
            del self._cache[key]
            return None
        return [dict(item) for item in entry["results"]]

    def _set_cache(self, key: str, results: list[dict[str, str]]) -> None:
        self._cache[key] = {
            "expires_at": time.time() + self.cache_ttl_seconds,
            "results": [dict(item) for item in results],
        }

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "SafetyAwarenessPulse/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            self._rate["last_error"] = detail
            if exc.code == 429:
                self._rate["limited"] = True
            raise RuntimeError(f"Serper HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            self._rate["last_error"] = str(exc)
            raise RuntimeError(f"Serper connection error: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Serper returned non-JSON: {raw[:200]}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected Serper payload type: {type(data).__name__}")
        return data

    @staticmethod
    def _parse_serper_payload(raw: dict[str, Any], *, news: bool) -> list[dict[str, str]]:
        parsed: list[dict[str, str]] = []
        if news:
            items = raw.get("news") or []
        else:
            items = raw.get("organic") or []
            # Prefer answer-box / knowledge graph snippets when organic is thin.
            answer = raw.get("answerBox") or {}
            if isinstance(answer, dict) and (answer.get("answer") or answer.get("snippet")):
                parsed.append(
                    {
                        "title": str(answer.get("title") or "Answer"),
                        "url": str(answer.get("link") or ""),
                        "content": str(answer.get("answer") or answer.get("snippet") or ""),
                    }
                )

        for item in items:
            if not isinstance(item, dict):
                continue
            parsed.append(
                {
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("link") or item.get("url") or ""),
                    "content": str(
                        item.get("snippet")
                        or item.get("description")
                        or item.get("content")
                        or ""
                    ),
                }
            )
        return [item for item in parsed if item.get("title") or item.get("content")]
