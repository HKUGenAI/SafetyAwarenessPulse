# Safety Awareness Pulse

A LangChain agent that posts a **daily workplace safety alert** in Traditional Chinese.

It first looks up working accidents that happened on the same month-day (`MM-DD`) in a local SQLite events database built from four Traditional Chinese PowerPoint newspaper-cutting decks. If nothing is found, it searches the [Hong Kong Labour Department press-release list](https://www.labour.gov.hk/tc/major/content.php), then falls back to worldwide workplace accidents, then historical facts.

## Architecture

| Layer | Choice |
| --- | --- |
| LLM | Azure OpenAI `gpt-5.4-mini` via [`langchain-openai`](https://pypi.org/project/langchain-openai/) (`AzureChatOpenAI`) |
| Agent | LangChain `create_agent` with three tools |
| Event store | Local **SQLite** (`data/events.db`); one row per accident; lookup by `month_day` |

| PPT parsing | `python-pptx` (original Traditional Chinese text is **not** translated) |
| Web search | [Serper](https://serper.dev/) Google Search API (`multi_search_api.SmartSearchTool`) |
| UI | Optional Streamlit local demo |
| Discord | Channel broadcast via `discord_broadcast.py` (09:00 HK short reminder + in-app details) |

## 4-level fallback (stop at the first hit)

1. **Local SQLite** — `WHERE month_day = MM-DD` against events ingested from the 4 PPTs.
2. **Labour Department press releases** — search [labour.gov.hk news](https://www.labour.gov.hk/tc/major/content.php) for workplace accidents on this day.
3. **World workplace web search** — lookup for working accidents worldwide on this day.
4. **Historical fact** — if no working accident is found, search for an interesting fact on this day in previous years.

All **user-facing** answers are Traditional Chinese. Retrieved PPT wording is kept in the original Traditional Chinese; it is never translated.

## Project layout

```
SafetyAwarenessPulse/
├── assets/ppts/              # Place the 4 Traditional Chinese PPT files here
├── data/events.db            # Created by document_ingest.py (gitignored)
├── deploy/                   # systemd examples for VPS
├── config.py
├── events_db.py              # SQLite schema + date lookups
├── document_ingest.py        # Load PPT slides → structured events → SQLite
├── multi_search_api.py       # Serper SmartSearchTool (cache + rate limits)
├── agent_mtr_bot.py          # Azure OpenAI agent + three tools + CLI chatbot
├── discord_broadcast.py      # Daily 09:00 HK Discord reminder + 閱讀全文
├── streamlit_app.py          # Optional local Traditional Chinese web demo
├── requirements.txt
├── .env.example
└── README.md
```

The four source decks already in `assets/ppts`:

- `2023 Newspaper Cutting.pptx`
- `2024 Newspaper Cutting.pptx`
- `2025 Newspaper Cutting.pptx`
- `2026 Newspaper Cutting.pptx`

## Prerequisites

- Python 3.10 or newer
- An [Azure OpenAI](https://oai.azure.com/) API key with a `gpt-5.4-mini` deployment
- A [Serper API key](https://serper.dev/)

## Setup

### 1. Create a virtual environment and install dependencies

Windows (PowerShell):

```powershell
cd D:\RA\SafetyAwarenessPulse
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
cd SafetyAwarenessPulse
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
If went wrong in Activating virtuel environement, try this command line before activating:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2. Configure API keys

```powershell
copy .env.example .env
```

Edit `.env`:

```
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_ENDPOINT=https://mtr-project.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-5.4-mini
SERPER_API_KEY=...
```

For Discord channel broadcast, also set:

```
DISCORD_BOT_TOKEN=...
DISCORD_CHANNEL_ID=123456789012345678
```

### 3. Confirm the PPT files are in place

Put (or keep) the four Traditional Chinese PPT files in:

```
./assets/ppts
```

### 4. Ingest documents into SQLite (run this first)

This reads every slide with `python-pptx`, keeps the original Traditional Chinese text, parses the event date, and writes **one row per accident** into `./data/events.db`. Same calendar day can have multiple rows.

```powershell
python document_ingest.py
```

Rebuild from scratch:

```powershell
python document_ingest.py --reset
```

Slides without a parseable date are skipped (they cannot be keyed by date).

### 5. Run the chatbot

Terminal agent (prints today's alert, then waits for questions):

```powershell
python agent_mtr_bot.py --chat
```

Generate only today's alert:

```powershell
python agent_mtr_bot.py
python agent_mtr_bot.py --date 2026-04-04
python agent_mtr_bot.py --chat
```

Streamlit UI (optional local demo):

```powershell
streamlit run streamlit_app.py
```

Deep link (local only):

```
http://localhost:8501/?date=2026-04-04
```

In the UI:

1. Pick a Hong Kong date in the sidebar and click **產生／重新產生警示**.
2. Read the daily safety alert (or historical fact).
3. Click **點擊此警示，繼續追問**.
4. Type follow-up questions in **想了解更多？在此提問**.

## Discord channel broadcast

Every day at **09:00 Asia/Hong_Kong**, Discord gets a short Traditional Chinese post:

1. One **news sentence** (date, place, what happened, and why), e.g. `2023年4月4日，港鐵灣仔站有維修工因未注意路軌環境跌倒受傷。`
2. One **safety tip**.

The **閱讀全文** button shows **date + that incident's news story** (original PPT wording when available). It does not show `[HITS]`, source-file metadata, fallback-level labels, or a list of search URLs.

### 1. Create the Discord bot

1. Open [Discord Developer Portal](https://discord.com/developers/applications) → **New Application** → **Bot** → copy `DISCORD_BOT_TOKEN`.
2. OAuth2 → URL Generator → scopes: `bot` → permissions: **Send Messages**, **Read Message History**, **Embed Links** (optional).
3. Invite the bot into your server with that URL.
4. Enable **Developer Mode** in Discord (Settings → Advanced) → right-click the target channel → **Copy Channel ID** → `DISCORD_CHANNEL_ID`.

### 2. Local test send

```powershell
pip install -r requirements.txt
# fill DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID in .env
python discord_broadcast.py --once
python discord_broadcast.py --once --date 2026-04-04
```

`--once` posts immediately, then **keeps the bot online** so **閱讀全文** still works. Stop with Ctrl+C.

### 3. Run the daily scheduler

```powershell
python discord_broadcast.py
```

Optional overrides in `.env`: `DISCORD_BROADCAST_HOUR`, `DISCORD_BROADCAST_MINUTE`.

Event details for each posted day are cached under `data/discord_detail_cache.json` so button clicks survive bot restarts.

## Deploy on a Linux server (Discord bot + tmux)

No public website is required. The bot only needs outbound HTTPS (Azure OpenAI, Serper, Discord) and a long-running process.

### 1. Upload the project from Windows

Connect to HKU lab hosts via Jump Host (example):

```powershell
ssh -L 9998:10.21.36.11:9998 -J h3629140@gatekeeper.cs.hku.hk qiyue@10.21.36.11
```

Upload from your PC (replace user/host; **do not upload Windows `.venv`**):

```powershell
ssh -J h3629140@gatekeeper.cs.hku.hk qiyue@10.21.36.11 "mkdir -p ~/SafetyAwarenessPulse"

cd D:\RA\SafetyAwarenessPulse
scp -J h3629140@gatekeeper.cs.hku.hk -r `
  agent_mtr_bot.py config.py discord_broadcast.py document_ingest.py events_db.py `
  multi_search_api.py streamlit_app.py requirements.txt .env.example README.md `
  assets data deploy `
  qiyue@10.21.36.11:~/SafetyAwarenessPulse/
```

Also upload `.env` (or create it on the server in the next step):

```powershell
scp -J h3629140@gatekeeper.cs.hku.hk D:\RA\SafetyAwarenessPulse\.env qiyue@10.21.36.11:~/SafetyAwarenessPulse/.env
```

Copy either `data/events.db` **or** `assets/ppts/` so the server can load / rebuild the SQLite store.

Gatekeeper may require **HKU network or HKUVPN**.

### 2. Create `.env` on the server

Option A — upload (above).  
Option B — edit on the server:

```bash
cd ~/SafetyAwarenessPulse
cp -n .env.example .env
nano .env
chmod 600 .env
```

Required keys:

```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://mtr-project.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-5.4-mini
SERPER_API_KEY=...
DISCORD_BOT_TOKEN=...
DISCORD_CHANNEL_ID=...
DISCORD_BROADCAST_HOUR=9
DISCORD_BROADCAST_MINUTE=30
```

### 3. Python environment (with or without sudo)

**If you have sudo** (normal Ubuntu/VPS):

```bash
sudo apt update
sudo apt install -y tmux python3-venv python3-pip
cd ~/SafetyAwarenessPulse
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**If you do NOT have sudo** (common on shared lab machines — `apt` / `python3 -m venv` fails):

```bash
cd ~/SafetyAwarenessPulse
rm -rf .venv

curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --user
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

python3 -m pip install --user virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If `data/events.db` is missing and PPTs are present:

```bash
python document_ingest.py --reset
```

Smoke-test once (bot stays online for the **閱讀全文** button):

```bash
python discord_broadcast.py --once
# Ctrl+C when done testing
```

### 4. Run under tmux (recommended)

```bash
cd ~/SafetyAwarenessPulse
tmux new -s safety-bot
```

Inside the green tmux bar session:

```bash
source .venv/bin/activate
python discord_broadcast.py
```

You should see login / `Scheduler started...` lines.

| Action | How |
| --- | --- |
| Detach (keep bot running) | `Ctrl+B`, then `D` |
| Re-attach after SSH | `tmux attach -t safety-bot` |
| List sessions | `tmux ls` |
| Stop bot | `tmux kill-session -t safety-bot` |

Optional helper script (needs a working `.venv` first):

```bash
chmod +x deploy/tmux_start.sh
./deploy/tmux_start.sh
tmux attach -t safety-bot
```

Stop any local Windows `python discord_broadcast.py` so you do not double-post.

### 5. Optional: systemd (machines with sudo)

For a VPS where you control root, see `deploy/safety-discord.service`:

```bash
# edit WorkingDirectory / User in the unit file first
sudo cp deploy/safety-discord.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now safety-discord
sudo systemctl status safety-discord
```

(Optional Streamlit UI: `deploy/nginx-safety-awareness.conf` + `deploy/safety-streamlit.service`. Discord details do not need it.)

## Tools

| Tool | Role |
| --- | --- |
| `search_local_work_accidents` | Query SQLite by `month_day` (`MM-DD`). If several accidents match, **one is chosen at random**. Returns original PPT text or `[NO_HITS]`. |
| `search_labour_department` | Search [Labour Department press releases](https://www.labour.gov.hk/tc/major/content.php) (`labour.gov.hk` only). Layer 2 after local DB. |
| `search_web` | Web search. Used for levels 3–4 and follow-up questions. Returns `[WEB_HITS]`, `[EMPTY_SEARCH]`, or `[API_ERROR]`. |

The Azure OpenAI model decides which tool to call. The daily-alert prompt forces this order: local SQLite first, then the Labour Department site, then world workplace search, then a historical fact.

## Date handling

- “Today” is computed in `Asia/Hong_Kong`.
- PPT cuttings in this dataset use **DD-MM-YYYY** (example: `04-04-2023`).
- Each stored accident is **one SQLite row** (`id` primary key). `iso_date` / `month_day` are indexed lookup fields (not unique).
- Daily alerts match by **month-day only** (`MM-DD` index), so 2023-04-04 can trigger an alert on 2026-04-04.
- If several accidents share the same `MM-DD`, the daily alert **randomly picks one**.
- The CLI / UI also accept `YYYY-MM-DD` and `MM-DD`.

## Query the SQLite database (`data/events.db`)

`data/events.db` is a local **SQLite** file. Table: `events`.

| Column | Meaning |
| --- | --- |
| `id` | Primary key |
| `iso_date` | Full date `YYYY-MM-DD` |
| `month_day` | `MM-DD` (same-day-in-history lookup) |
| `title` / `location` / `category` | Parsed fields |
| `content` | Original Traditional Chinese PPT text |
| `source_file` / `slide_number` | Source slide |

### PowerShell one-liners

Use parameterized `?` placeholders so quotes stay simple:

```powershell
# Count rows
python -c "import sqlite3; c=sqlite3.connect(r'data\events.db'); print(list(c.execute('SELECT COUNT(*) FROM events')))"

# Same month-day as the bot (e.g. 04-04)
python -c "import sqlite3; c=sqlite3.connect(r'data\events.db'); print(list(c.execute('SELECT id, iso_date, title FROM events WHERE month_day = ?', ('04-04',))))"

# Exact calendar date (zero-padded YYYY-MM-DD)
python -c "import sqlite3; c=sqlite3.connect(r'data\events.db'); print(list(c.execute('SELECT id, title, location FROM events WHERE iso_date = ?', ('2023-04-04',))))"

# Keyword search in title / body
python -c "import sqlite3; c=sqlite3.connect(r'data\events.db'); print(list(c.execute('SELECT id, iso_date, title FROM events WHERE content LIKE ? OR title LIKE ?', ('%港鐵%', '%港鐵%'))))"
```

### Interactive `sqlite3` CLI (if installed)

```powershell
sqlite3 data\events.db
```

```sql
.headers on
.mode column

SELECT COUNT(*) FROM events;

SELECT id, iso_date, title, location
FROM events
WHERE month_day = '04-04';

SELECT id, title, location
FROM events
WHERE iso_date = '2023-04-04';

SELECT content FROM events WHERE id = 1;
```

Type `.quit` to exit.

### Helpers in `events_db.py`

```python
from events_db import fetch_by_month_day, fetch_by_iso_date

print(fetch_by_month_day("04-04"))
print(fetch_by_iso_date("2023-04-04"))
```

## Error handling

| Situation | Behaviour |
| --- | --- |
| `data/events.db` missing | Local tool returns an error telling you to run `document_ingest.py` |
| No accident hit on that `MM-DD` | `[NO_HITS]` → agent moves to the next fallback level |
| Empty web result | `[EMPTY_SEARCH]` → agent moves to the next fallback level |
| Azure OpenAI / web API connection error | User-facing Traditional Chinese error; the app does not crash |

## Testing guide (4-level fallback)

The PPTs are industrial-accident newspaper cuttings (construction, factory, railway, and other workplace cases). Example records (DD-MM-YYYY):

- `29-12-2023` — 元朗工業邨燒焊金屬鐡擊中工人
- `04-04-2023` — 港鐵灣仔站維修工於路軌跌倒
- `01-03-2023` — 港鐵旺角站扶手梯維修

### Level 1 — local PPT / SQLite

```powershell
python agent_mtr_bot.py --date 2026-12-29
python agent_mtr_bot.py --date 2026-04-04
```

Expect:

- Tool trace contains **only** `search_local_work_accidents`
- The notice quotes original Traditional Chinese PPT wording
- No web call

In Streamlit, set the sidebar date to a PPT accident date and open **工具呼叫紀錄**.

### Level 2 — Labour Department press releases

Pick an `MM-DD` that has **no** accident slide in the four PPTs, but may appear on [labour.gov.hk news](https://www.labour.gov.hk/tc/major/content.php) (for example a recent fatal workplace accident date):

```powershell
python agent_mtr_bot.py --date YYYY-MM-DD --test-fallback
```

Expect level 1 to return `[NO_HITS]`, then `search_labour_department` to query the official listing first.

### Level 3 — world workplace web search

If the Labour Department listing also has no workplace accident for that `MM-DD`, the next `search_web` query must be a **global** workplace / industrial accident search for the same day. Check the tool trace in the terminal or Streamlit expander.

### Level 4 — historical fact

If levels 1–3 all miss, the last query is `historical events` / 歷史上的今天. The UI still answers in Traditional Chinese, but the card is a fact rather than a workplace notice. Click it and ask follow-up questions as usual.

### Error-path checks

1. Rename `data/events.db` and call the agent → ingest error, no crash.
2. Put an invalid `SERPER_API_KEY` in `.env` → `[API_ERROR]` from `search_web`.
3. After an alert is shown, ask：`這次意外的主要風險是甚麼？` → Traditional Chinese follow-up, tools optional.

## Notes

- PPT content is industrial-accident newspaper cuttings. Level 1 accepts **any** working accident on that `MM-DD`, not only railway cases.
- The events DB lives entirely on disk under `data/events.db`. No vector-DB cloud is used.
- LLM calls go to Azure OpenAI deployment `gpt-5.4-mini` (`AZURE_OPENAI_DEPLOYMENT`).
