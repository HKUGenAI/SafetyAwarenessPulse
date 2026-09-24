"""
Simple Streamlit demo for the daily workplace safety-alert agent.

Sidebar language switch: Traditional Chinese or English.
ChromaDB / retrieval logic is unchanged.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

# Must be set before importing streamlit so the file watcher never scans transformers.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st
from dotenv import load_dotenv

from config import HONG_KONG_TZ, PPT_DIR

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

UI = {
    "zh-Hant": {
        "page_title": "每日安全警示",
        "title": "⚠️ 每日安全警示",
        "caption": "以史為鑑　·　本地 PPT 知識庫優先　·　工作意外安全提示",
        "settings": "設定",
        "language": "介面語言",
        "missing_deepseek": "尚未設定 DEEPSEEK_API_KEY。請複製 .env.example 為 .env。",
        "missing_serper": "尚未設定 SERPER_API_KEY。請複製 .env.example 為 .env。",
        "alert_date": "警示日期（香港）",
        "date_hint": "將使用月日 **{month_day}** 搜尋往年同日事件。",
        "generate": "重新產生警示",
        "generating": "正在載入模型並根據四層備援規則產生每日安全警示…",
        "ppt_sources": "資料來源 PPT",
        "ppt_missing": "找不到 PPT。請把 4 份繁體中文簡報放到 `assets/ppts`。",
        "fallback": "**備援順序**\n1. 本地 RAG（4 份 PPT 工作意外）\n2. 勞工處新聞公報（labour.gov.hk）\n3. 網上全球工作意外\n4. 歷史上的今天",
        "pick_date": "正在產生今日警示，請稍候。",
        "today": "今日提示",
        "default_level": "安全警示",
        "system_error": "系統錯誤：{error}",
        "click_alert": "點擊此警示，繼續追問",
        "clicked": "已載入此警示作為對話背景。請在右側提問。",
        "tool_trace": "工具呼叫紀錄（用於驗證四層備援）",
        "no_tools": "沒有工具呼叫紀錄。",
        "ask_more": "想了解更多？在此提問",
        "ask_hint": "先點擊左側警示，或直接在下方輸入問題。",
        "thinking": "助手正在思考並視需要呼叫工具…",
        "tools_prefix": "工具：",
        "lang_zh": "繁體中文",
        "lang_en": "English",
    },
    "en": {
        "page_title": "Daily Safety Alert",
        "title": "⚠️ Daily Safety Alert",
        "caption": "Learn from history · Local PPT knowledge base first · Workplace safety notice",
        "settings": "Settings",
        "language": "Interface language",
        "missing_deepseek": "DEEPSEEK_API_KEY is not set. Copy .env.example to .env.",
        "missing_serper": "SERPER_API_KEY is not set. Copy .env.example to .env.",
        "alert_date": "Alert date (Hong Kong)",
        "date_hint": "Will search previous years for the same month-day **{month_day}**.",
        "generate": "Regenerate alert",
        "generating": "Loading the model and generating the daily safety alert with the 4-level fallback…",
        "ppt_sources": "Source PPT files",
        "ppt_missing": "No PPT files found. Place the 4 Traditional Chinese decks in `assets/ppts`.",
        "fallback": "**Fallback order**\n1. Local RAG (workplace accidents in the 4 PPTs)\n2. Labour Department press releases (labour.gov.hk)\n3. Worldwide workplace accidents on the web\n4. On this day in history",
        "pick_date": "Generating today’s alert, please wait.",
        "today": "Today’s notice",
        "default_level": "Safety alert",
        "system_error": "System error: {error}",
        "click_alert": "Click this alert to ask follow-up questions",
        "clicked": "This alert is now the chat context. Ask a question on the right.",
        "tool_trace": "Tool-call log (for verifying the 4-level fallback)",
        "no_tools": "No tool calls recorded.",
        "ask_more": "Want to know more? Ask here",
        "ask_hint": "Click the alert on the left, or type a question below.",
        "thinking": "The assistant is thinking and will call tools if needed…",
        "tools_prefix": "Tools: ",
        "lang_zh": "繁體中文",
        "lang_en": "English",
    },
}

st.set_page_config(
    page_title="每日安全警示 / Daily Safety Alert",
    page_icon="⚠️",
    layout="wide",
)

st.markdown(
    """
    <style>
      html, body, [class*="css"]  {
        font-family: "Microsoft JhengHei", "PingFang TC", "Noto Sans TC", "Segoe UI", sans-serif;
      }
      .alert-card {
        border: 1px solid #f3c4c4;
        background: #fff7f7;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
      }
      .level-chip {
        display: inline-block;
        background: #b42318;
        color: #fff;
        border-radius: 999px;
        padding: 0.15rem 0.75rem;
        font-size: 0.85rem;
        margin-bottom: 0.6rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _agent_mod():
    """Load the LangChain agent only when the user actually needs it."""
    import agent_mtr_bot as bot

    return bot


def _init_state() -> None:
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "alert" not in st.session_state:
        st.session_state.alert = None
    if "alert_clicked" not in st.session_state:
        st.session_state.alert_clicked = False
    if "alert_key" not in st.session_state:
        st.session_state.alert_key = None
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "zh-Hant"


def _history_for_agent(language: str) -> list[dict[str, str]]:
    bot = _agent_mod()
    history: list[dict[str, str]] = []
    alert = st.session_state.alert
    if alert:
        target = date.fromisoformat(alert["iso_date"])
        history.append({"role": "user", "content": bot.daily_alert_user_prompt(target, language=language)})
        history.append({"role": "assistant", "content": alert["text"]})
    for turn in st.session_state.chat:
        if turn["role"] == "user":
            history.append({"role": "user", "content": bot.wrap_user_message(turn["content"], language)})
        else:
            history.append({"role": turn["role"], "content": turn["content"]})
    return history


def generate_and_store(target: date, language: str) -> None:
    copy = UI[language]
    with st.spinner(copy["generating"]):
        bot = _agent_mod()
        st.session_state.alert = bot.generate_daily_alert(target, language=language)
        st.session_state.chat = []
        st.session_state.alert_clicked = False
        st.session_state.alert_key = f"{target.isoformat()}|{language}"


_init_state()

lang_choice = st.sidebar.radio(
    UI[st.session_state.ui_lang]["language"],
    options=["zh-Hant", "en"],
    format_func=lambda code: UI["zh-Hant"]["lang_zh"] if code == "zh-Hant" else UI["en"]["lang_en"],
    key="ui_lang",
)
language = lang_choice
t = UI[language]

st.title(t["title"])
st.caption(t["caption"])

with st.sidebar:
    st.header(t["settings"])
    if not os.getenv("DEEPSEEK_API_KEY"):
        st.error(t["missing_deepseek"])
    if not os.getenv("SERPER_API_KEY"):
        st.error(t["missing_serper"])
    default_day = datetime.now(ZoneInfo(HONG_KONG_TZ)).date()
    selected_day = st.date_input(t["alert_date"], value=default_day, format="YYYY-MM-DD")
    st.caption(t["date_hint"].format(month_day=selected_day.strftime("%m-%d")))

    alert_key = f"{selected_day.isoformat()}|{language}"
    can_call_llm = bool(os.getenv("DEEPSEEK_API_KEY"))
    if can_call_llm and st.session_state.alert_key != alert_key:
        generate_and_store(selected_day, language)
    if st.button(t["generate"], use_container_width=True) and can_call_llm:
        generate_and_store(selected_day, language)

    st.divider()
    st.markdown(f"**{t['ppt_sources']}**")
    ppt_files = sorted(PPT_DIR.glob("*.pptx"))
    if ppt_files:
        for path in ppt_files:
            st.write(f"- {path.name}")
    else:
        st.warning(t["ppt_missing"])

    st.divider()
    st.markdown(t["fallback"])

if st.session_state.alert is None:
    st.info(t["pick_date"])
    st.stop()

alert = st.session_state.alert
level_key = alert.get("level_key") or "unknown"
bot_mod = _agent_mod()
level_label = bot_mod.LEVEL_LABELS.get(level_key, bot_mod.LEVEL_LABELS["unknown"]).get(
    language, t["default_level"]
)

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader(t["today"])
    st.markdown(
        f'<div class="alert-card"><div class="level-chip">{level_label}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(alert["text"])

    if alert.get("error"):
        st.error(t["system_error"].format(error=alert["error"]))

    clicked = st.button(t["click_alert"], use_container_width=True)
    if clicked:
        st.session_state.alert_clicked = True
        st.success(t["clicked"])

    with st.expander(t["tool_trace"], expanded=False):
        trace = alert.get("trace") or []
        if trace:
            for step in trace:
                st.code(step, language="text")
        else:
            st.write(t["no_tools"])

with right:
    st.subheader(t["ask_more"])
    if not st.session_state.alert_clicked and not st.session_state.chat:
        st.caption(t["ask_hint"])

    for turn in st.session_state.chat:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.markdown(turn["content"])

    question = st.chat_input(t["ask_more"])
    if question:
        bot_mod = _agent_mod()
        st.session_state.alert_clicked = True
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner(t["thinking"]):
                reply = bot_mod.invoke_agent(
                    bot_mod.wrap_user_message(question, language),
                    history=_history_for_agent(language)[:-1],
                    language=language,
                )
            st.markdown(reply["text"])
            if reply.get("trace"):
                st.caption(t["tools_prefix"] + " → ".join(reply["trace"]))
        st.session_state.chat.append({"role": "assistant", "content": reply["text"]})
