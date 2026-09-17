# Safety Awareness Pulse

A LangChain agent that posts a **daily workplace safety alert** in Traditional Chinese.

It first looks up working accidents that happened on the same month-day (`MM-DD`) in a local RAG knowledge base built from four Traditional Chinese PowerPoint newspaper-cutting decks. If nothing is found, it searches the [Hong Kong Labour Department press-release list](https://www.labour.gov.hk/tc/major/content.php), then falls back to worldwide workplace accidents, then historical facts.

## Architecture

| Layer | Choice |
| --- | --- |
| LLM | `deepseek-chat` via [`langchain-deepseek`](https://pypi.org/project/langchain-deepseek/) (`ChatDeepSeek`, DeepSeek function calling) |
| Agent | LangChain `create_agent` with three tools |
| Vector store | Local [ChromaDB](https://www.trychroma.com/) (no cloud) |
| Embeddings | Local Hugging Face model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (DeepSeek has no public embedding API) |
| PPT parsing | `python-pptx` (original Traditional Chinese text is **not** translated) |
| Web search | Tavily Search API (`langchain-tavily`) |
| UI | Streamlit (Traditional Chinese copy) |

## 4-level fallback (stop at the first hit)

1. **Local RAG** — search the 4 PPTs in ChromaDB for working accidents on the same `MM-DD`.
2. **Labour Department press releases** — search [labour.gov.hk news](https://www.labour.gov.hk/tc/major/content.php) for workplace accidents on this day.
3. **World workplace web search** — Tavily lookup for working accidents worldwide on this day.
4. **Historical fact** — if no working accident is found, search for an interesting fact on this day in previous years.

All **user-facing** answers are Traditional Chinese. Retrieved PPT wording is kept in the original Traditional Chinese; it is never translated before embedding or after retrieval.

## Project layout

```
SafetyAwarenessPulse/
├── assets/ppts/              # Place the 4 Traditional Chinese PPT files here
├── chroma_db/                # Created by document_ingest.py (gitignored)
├── config.py
├── document_ingest.py        # Load → parse → chunk → embed → ChromaDB
├── agent_mtr_bot.py          # DeepSeek agent + two tools + CLI chatbot
├── streamlit_app.py          # Traditional Chinese web demo
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
- A [DeepSeek API key](https://platform.deepseek.com/)
- A [Tavily API key](https://tavily.com/)
- Disk space for the first-time download of the local embedding model (~100 MB)

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
DEEPSEEK_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 3. Confirm the PPT files are in place

Put (or keep) the four Traditional Chinese PPT files in:

```
./assets/ppts
```

### 4. Ingest documents into local ChromaDB (run this first)

This reads every slide with `python-pptx`, keeps the original Traditional Chinese text, splits long slides, embeds them locally, and writes `./chroma_db`.

```powershell
python document_ingest.py
```

Rebuild from scratch:

```powershell
python document_ingest.py --reset
```

The first run downloads the embedding model. Ingesting four large decks on CPU can take several minutes.

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

Streamlit UI (Traditional Chinese):

```powershell
streamlit run streamlit_app.py
```

In the UI:

1. Pick a Hong Kong date in the sidebar and click **產生／重新產生警示**.
2. Read the daily safety alert (or historical fact).
3. Click **點擊此警示，繼續追問**.
4. Type follow-up questions in **想了解更多？在此提問**.

## Tools

| Tool | Role |
| --- | --- |
| `search_local_work_accidents` | Query ChromaDB. Matches `month_day` metadata (`MM-DD`) for any workplace accident in the 4 PPTs. Returns original PPT text or `[NO_HITS]`. |
| `search_labour_department` | Search [Labour Department press releases](https://www.labour.gov.hk/tc/major/content.php) (`labour.gov.hk` only). Layer 2 after local RAG. |
| `search_web` | Tavily Search. Used for levels 3–4 and follow-up questions. Returns `[WEB_HITS]`, `[EMPTY_SEARCH]`, or `[API_ERROR]`. |

The DeepSeek model decides which tool to call. The daily-alert prompt forces this order: local RAG first, then the Labour Department site, then world workplace search, then a historical fact.

## Date handling

- “Today” is computed in `Asia/Hong_Kong`.
- PPT cuttings in this dataset use **DD-MM-YYYY** (example: `04-04-2023`).
- Matching is by **month-day only** (`MM-DD`), so 2023-04-04 can trigger an alert on 2026-04-04.
- The CLI / UI also accept `YYYY-MM-DD` and `MM-DD`.

## Error handling

| Situation | Behaviour |
| --- | --- |
| ChromaDB folder missing | RAG tool returns an error telling you to run `document_ingest.py` |
| No accident hit on that `MM-DD` | `[NO_HITS]` → agent moves to the next fallback level |
| Empty Tavily result | `[EMPTY_SEARCH]` → agent moves to the next fallback level |
| DeepSeek / Tavily connection error | User-facing Traditional Chinese error; the app does not crash |

## Testing guide (4-level fallback)

The PPTs are industrial-accident newspaper cuttings (construction, factory, railway, and other workplace cases). Example records (DD-MM-YYYY):

- `29-12-2023` — 元朗工業邨燒焊金屬鐡擊中工人
- `04-04-2023` — 港鐵灣仔站維修工於路軌跌倒
- `01-03-2023` — 港鐵旺角站扶手梯維修

### Level 1 — local PPT / RAG

```powershell
python agent_mtr_bot.py --date 2026-12-29
python agent_mtr_bot.py --date 2026-04-04
```

Expect:

- Tool trace contains **only** `search_local_work_accidents`
- The notice quotes original Traditional Chinese PPT wording
- No Tavily call

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

1. Rename `chroma_db` and call the agent → ingest error, no crash.
2. Put an invalid `TAVILY_API_KEY` in `.env` → `[API_ERROR]` from `search_web`.
3. After an alert is shown, ask：`這次意外的主要風險是甚麼？` → Traditional Chinese follow-up, tools optional.

## Notes

- PPT content is industrial-accident newspaper cuttings. Level 1 accepts **any** working accident on that `MM-DD`, not only railway cases.
- Embeddings and Chroma live entirely on disk under `chroma_db/`. Nothing is sent to a vector-DB cloud.
- `deepseek-reasoner` is **not** used: DeepSeek documents function calling on `deepseek-chat`.
