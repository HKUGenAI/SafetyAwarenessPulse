"""
Discord broadcaster for the daily short safety reminder.

Posts one Traditional Chinese 30–50 character reminder to a Discord channel
every day at 09:00 Asia/Hong_Kong (configurable), with an "了解更多" button
that reveals the full event details inside Discord (no external website).

Usage:
    python discord_broadcast.py --once          # send now, then keep bot online
    python discord_broadcast.py                 # daily scheduler + button handler
    python discord_broadcast.py --date 2026-04-04 --once
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import discord
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from agent_mtr_bot import generate_short_reminder, today_in_hong_kong
from config import DATA_DIR, DISCORD_BROADCAST_HOUR, DISCORD_BROADCAST_MINUTE, HONG_KONG_TZ

DETAIL_CACHE_PATH = DATA_DIR / "discord_detail_cache.json"
CUSTOM_ID_PREFIX = "safety_detail:"
DISCORD_MSG_LIMIT = 1900


def _broadcast_hour() -> int:
    return int(os.getenv("DISCORD_BROADCAST_HOUR", str(DISCORD_BROADCAST_HOUR)))


def _broadcast_minute() -> int:
    return int(os.getenv("DISCORD_BROADCAST_MINUTE", str(DISCORD_BROADCAST_MINUTE)))


def _require_settings() -> tuple[str, int]:
    token = (os.getenv("DISCORD_BOT_TOKEN") or "").strip()
    channel_raw = (os.getenv("DISCORD_CHANNEL_ID") or "").strip()
    missing = []
    if not token or token.startswith("your-"):
        missing.append("DISCORD_BOT_TOKEN")
    if not channel_raw:
        missing.append("DISCORD_CHANNEL_ID")
    if missing:
        raise RuntimeError(
            "Missing required settings: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )
    try:
        channel_id = int(channel_raw)
    except ValueError as exc:
        raise RuntimeError("DISCORD_CHANNEL_ID must be a numeric snowflake id.") from exc
    return token, channel_id


def _load_detail_cache() -> dict[str, str]:
    if not DETAIL_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(DETAIL_CACHE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_detail_cache(cache: dict[str, str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Keep only the latest ~60 days of details to avoid unbounded growth.
    keys = sorted(cache.keys())
    if len(keys) > 60:
        cache = {k: cache[k] for k in keys[-60:]}
    DETAIL_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def cache_detail(iso_date: str, detail_text: str) -> None:
    cache = _load_detail_cache()
    cache[iso_date] = detail_text
    _save_detail_cache(cache)


def get_cached_detail(iso_date: str) -> str | None:
    return _load_detail_cache().get(iso_date)


def split_discord_chunks(text: str, limit: int = DISCORD_MSG_LIMIT) -> list[str]:
    text = (text or "").strip() or "（暫無詳細資料）"
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    return chunks


def build_message_text(reminder: dict) -> str:
    body = (reminder.get("text") or "").strip()
    return f"【每日安全提醒】\n{body}"


class DetailButton(discord.ui.Button):
    """Label-only button; clicks are handled in SafetyBot.on_interaction."""

    def __init__(self, iso_date: str):
        super().__init__(
            label="了解更多",
            style=discord.ButtonStyle.primary,
            custom_id=f"{CUSTOM_ID_PREFIX}{iso_date}",
        )


class DetailView(discord.ui.View):
    """Attached to each daily post. Timeout=None so the button stays usable."""

    def __init__(self, iso_date: str):
        super().__init__(timeout=None)
        self.add_item(DetailButton(iso_date))


async def _send_detail_response(interaction: discord.Interaction, iso_date: str) -> None:
    detail = get_cached_detail(iso_date)
    if not detail:
        # Fallback: regenerate for that calendar day if cache was wiped.
        try:
            reminder = await asyncio.to_thread(
                generate_short_reminder,
                date.fromisoformat(iso_date),
            )
            detail = reminder.get("detail_text") or "（暫無詳細資料）"
            cache_detail(iso_date, detail)
        except Exception as exc:  # noqa: BLE001
            detail = f"無法載入 {iso_date} 的事件詳情：{exc}"

    chunks = split_discord_chunks(detail)
    await interaction.response.send_message(chunks[0], ephemeral=True)
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk, ephemeral=True)


class SafetyBot(discord.Client):
    def __init__(self, *, channel_id: int, once_target: date | None = None):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.channel_id = channel_id
        self.once_target = once_target
        self.scheduler: AsyncIOScheduler | None = None
        self._posted_once = False

    async def setup_hook(self) -> None:
        # Dynamic custom_ids (per iso_date) are handled in on_interaction.
        return

    async def on_ready(self) -> None:
        print(f"[discord] Logged in as {self.user} (id={self.user and self.user.id})")
        if self.once_target is not None and not self._posted_once:
            self._posted_once = True
            try:
                await self.post_reminder(self.once_target)
            except Exception as exc:  # noqa: BLE001
                print(f"[discord] ERROR posting once: {exc}", file=sys.stderr)
            return

        if self.scheduler is None:
            await self._start_scheduler()

    async def on_interaction(self, interaction: discord.Interaction) -> None:
        if interaction.type is not discord.InteractionType.component:
            return
        custom_id = ""
        if interaction.data and isinstance(interaction.data, dict):
            custom_id = str(interaction.data.get("custom_id") or "")
        if not custom_id.startswith(CUSTOM_ID_PREFIX):
            return
        if interaction.response.is_done():
            return
        iso_date = custom_id[len(CUSTOM_ID_PREFIX) :]
        await _send_detail_response(interaction, iso_date)

    async def _start_scheduler(self) -> None:
        hour = _broadcast_hour()
        minute = _broadcast_minute()
        tz = ZoneInfo(HONG_KONG_TZ)
        self.scheduler = AsyncIOScheduler(timezone=tz)
        self.scheduler.add_job(
            self._scheduled_post,
            CronTrigger(hour=hour, minute=minute, timezone=tz),
            id="daily_safety_reminder",
            replace_existing=True,
        )
        self.scheduler.start()
        print(
            f"[discord] Scheduler started. Posts daily at "
            f"{hour:02d}:{minute:02d} {HONG_KONG_TZ}."
        )
        print(f"[discord] Channel id={self.channel_id}")

    async def _scheduled_post(self) -> None:
        try:
            await self.post_reminder(today_in_hong_kong())
        except Exception as exc:  # noqa: BLE001
            print(f"[discord] ERROR during scheduled broadcast: {exc}", file=sys.stderr)

    async def post_reminder(self, target: date) -> dict:
        reminder = await asyncio.to_thread(generate_short_reminder, target)
        cache_detail(reminder["iso_date"], reminder.get("detail_text") or "")

        channel = self.get_channel(self.channel_id)
        if channel is None:
            channel = await self.fetch_channel(self.channel_id)
        if not isinstance(channel, discord.abc.Messageable):
            raise RuntimeError(f"Channel {self.channel_id} is not messageable.")

        text = build_message_text(reminder)
        view = DetailView(reminder["iso_date"])
        message = await channel.send(content=text, view=view)

        print("[discord] Posted to channel.")
        print(f"[discord] date={reminder['iso_date']} chars={reminder['char_count']}")
        print(f"[discord] text={reminder['text']}")
        print(f"[discord] message_id={message.id}")
        if reminder.get("error"):
            print(f"[discord] WARNING: LLM fallback used ({reminder['error']})")
        return {"message_id": message.id, "reminder": reminder}


def main() -> None:
    parser = argparse.ArgumentParser(description="Discord channel daily safety reminder.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Send one reminder now (bot stays online so 了解更多 keeps working).",
    )
    parser.add_argument(
        "--date",
        help="Override Hong Kong date for --once, YYYY-MM-DD or MM-DD.",
    )
    args = parser.parse_args()

    if args.date and not args.once:
        parser.error("--date is only valid together with --once")

    token, channel_id = _require_settings()
    once_target: date | None = None
    if args.once:
        once_target = today_in_hong_kong(args.date) if args.date else today_in_hong_kong()

    bot = SafetyBot(channel_id=channel_id, once_target=once_target)
    bot.run(token)


if __name__ == "__main__":
    main()
