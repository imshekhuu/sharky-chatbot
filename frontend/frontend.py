from __future__ import annotations

import os
import sys
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.backend import (
    chatbot,
    generate_chat_title,
    ingest_pdf,
    thread_document_metadata,
    thread_has_document,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sharky AI",
    page_icon="🦈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══ RESET ═══ */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background: #07090F !important;
    color: #D4DBE8;
}

/* ═══ SIDEBAR ═══ */
[data-testid="stSidebar"] {
    background: #0C1022 !important;
    border-right: 1px solid #161E35 !important;
    width: 272px !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

/* brand */
.s-brand {
    padding: 18px 16px 14px;
    border-bottom: 1px solid #161E35;
    display: flex; align-items: center; gap: 11px;
    margin-bottom: 4px;
}
.s-brand-icon {
    width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, #1742C8 0%, #0EA5E9 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.s-brand-text .name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px; font-weight: 500;
    color: #E8EEF8; letter-spacing: 0.05em;
    line-height: 1.2;
}
.s-brand-text .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #3B82F6; letter-spacing: 0.1em;
    margin-top: 2px;
}

/* new chat btn */
.stButton > button {
    width: 100% !important;
    background: #111827 !important;
    border: 1px solid #1F2D48 !important;
    color: #94A3B8 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: all 0.15s !important;
    letter-spacing: 0.02em;
    text-align: left !important;
}
.stButton > button:hover {
    background: #1A2640 !important;
    border-color: #2E4BCC !important;
    color: #CBD5E1 !important;
    box-shadow: 0 0 0 1px rgba(46,75,204,0.3) !important;
    transform: none !important;
}

/* section labels */
.s-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 500;
    letter-spacing: 0.14em; color: #2A3A5C;
    text-transform: uppercase;
    padding: 14px 16px 5px;
}

/* thread items */
.thread-item {
    display: flex; align-items: center; gap: 8px;
    padding: 7px 14px; border-radius: 0;
    cursor: pointer; transition: background 0.1s;
    border-left: 2px solid transparent;
    margin: 1px 0;
}
.thread-item:hover { background: #111827; }
.thread-item.active {
    background: #111827;
    border-left-color: #2E4BCC;
}
.thread-icon { font-size: 12px; opacity: 0.4; flex-shrink: 0; }
.thread-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #64748B;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.thread-item.active .thread-title { color: #94A3B8; }

/* doc badge in sidebar */
.doc-badge {
    margin: 8px 12px;
    background: rgba(14,165,233,0.06);
    border: 1px solid rgba(14,165,233,0.15);
    border-radius: 8px;
    padding: 9px 12px;
    display: flex; align-items: flex-start; gap: 9px;
}
.doc-badge-icon { font-size: 14px; margin-top: 1px; flex-shrink: 0; }
.doc-badge-body .doc-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #38BDF8; font-weight: 500;
    word-break: break-all; line-height: 1.4;
}
.doc-badge-body .doc-meta {
    font-size: 10px; color: #475569; margin-top: 2px;
}

/* tools status bar */
.tools-bar {
    position: fixed; bottom: 0; left: 0; width: 272px;
    background: #0C1022;
    border-top: 1px solid #161E35;
    padding: 10px 16px;
    display: flex; gap: 8px; flex-wrap: wrap;
    z-index: 100;
}
.tool-dot {
    display: flex; align-items: center; gap: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #334155; letter-spacing: 0.05em;
}
.tool-dot::before {
    content: ''; width: 5px; height: 5px;
    border-radius: 50%; background: #1D4ED8; flex-shrink: 0;
}
.tool-dot.active::before { background: #38BDF8; }
.tool-dot.active { color: #64748B; }

/* ═══ MAIN AREA ═══ */
.block-container { padding: 0 !important; max-width: 100% !important; }

/* header */
.chat-header {
    position: sticky; top: 0; z-index: 50;
    background: rgba(7,9,15,0.92);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #0F1729;
    padding: 13px 28px;
    display: flex; align-items: center; gap: 12px;
}
.chat-header-title {
    font-size: 14px; font-weight: 500; color: #CBD5E1;
    flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.live-dot {
    display: flex; align-items: center; gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #22C55E; letter-spacing: 0.08em;
}
.live-dot::before {
    content: ''; width: 5px; height: 5px;
    border-radius: 50%; background: #22C55E;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ═══ MESSAGES ═══ */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 6px 0 !important;
    max-width: 820px;
    margin: 0 auto !important;
}

/* avatars */
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #1742C8, #0EA5E9) !important;
    border: none !important;
    font-size: 14px !important;
}
[data-testid="chatAvatarIcon-user"] {
    background: #111827 !important;
    border: 1px solid #1F2D48 !important;
}

/* message text */
[data-testid="stChatMessage"] p {
    color: #C4CFDE !important;
    font-size: 14px !important;
    line-height: 1.75 !important;
    margin: 0 0 10px 0 !important;
}
[data-testid="stChatMessage"] p:last-child { margin-bottom: 0 !important; }

/* code blocks */
[data-testid="stChatMessage"] code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    background: #0C1022 !important;
    border: 1px solid #161E35 !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
    color: #38BDF8 !important;
}
[data-testid="stChatMessage"] pre {
    background: #0C1022 !important;
    border: 1px solid #161E35 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    overflow-x: auto !important;
}
[data-testid="stChatMessage"] pre code {
    border: none !important; background: transparent !important;
    padding: 0 !important; color: #94A3B8 !important;
}

/* tool status */
[data-testid="stStatus"] {
    background: #0C1022 !important;
    border: 1px solid #161E35 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    max-width: 340px;
}
[data-testid="stStatus"] summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important; color: #64748B !important;
}

/* ═══ INPUT AREA ═══ */
.input-wrapper {
    position: sticky; bottom: 0;
    background: rgba(7,9,15,0.95);
    backdrop-filter: blur(12px);
    border-top: 1px solid #0F1729;
    padding: 14px 28px 18px;
}
[data-testid="stChatInput"] {
    background: #0C1022 !important;
    border: 1px solid #1A2640 !important;
    border-radius: 12px !important;
    max-width: 820px !important;
    margin: 0 auto !important;
    transition: border-color 0.15s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #2E4BCC !important;
    box-shadow: 0 0 0 3px rgba(46,75,204,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #D4DBE8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    caret-color: #38BDF8 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #2A3A5C !important; }
[data-testid="stChatInput"] button {
    background: #1742C8 !important; border-radius: 8px !important;
}
[data-testid="stChatInput"] button:hover {
    background: #2E4BCC !important;
}

/* ═══ FILE UPLOADER ═══ */
[data-testid="stFileUploader"] {
    background: #0C1022 !important;
    border: 1px dashed #1F2D48 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"] label {
    color: #475569 !important;
    font-size: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stFileUploader"] button {
    background: #111827 !important;
    border: 1px solid #1F2D48 !important;
    color: #64748B !important;
    font-size: 11px !important;
    border-radius: 6px !important;
}

/* ═══ EMPTY STATE ═══ */
.empty-state {
    max-width: 440px; margin: 70px auto 0;
    text-align: center; padding: 0 24px;
}
.empty-fin { font-size: 44px; opacity: 0.18; margin-bottom: 20px; }
.empty-heading {
    font-size: 22px; font-weight: 500; color: #8896A8;
    margin-bottom: 10px; letter-spacing: -0.01em;
}
.empty-sub {
    font-size: 13px; color: #2A3A5C; line-height: 1.7;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 24px;
}
.cap-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px; text-align: left; margin-bottom: 28px;
}
.cap-card {
    background: #0C1022;
    border: 1px solid #161E35;
    border-radius: 10px; padding: 12px 14px;
}
.cap-card-icon { font-size: 16px; margin-bottom: 6px; }
.cap-card-title {
    font-size: 11px; font-weight: 600; color: #64748B;
    letter-spacing: 0.04em; font-family: 'JetBrains Mono', monospace;
    margin-bottom: 3px;
}
.cap-card-desc { font-size: 11px; color: #2A3A5C; line-height: 1.5; }

/* ═══ INFO CHIPS ═══ */
.chip {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 500; letter-spacing: 0.05em;
    border-radius: 20px; padding: 3px 10px;
}
.chip-blue {
    color: #38BDF8; background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.15);
}
.chip-green {
    color: #22C55E; background: rgba(34,197,94,0.07);
    border: 1px solid rgba(34,197,94,0.15);
}
.chip-amber {
    color: #F59E0B; background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.15);
}

/* ═══ INGEST SUCCESS ALERT ═══ */
.ingest-success {
    background: rgba(34,197,94,0.06);
    border: 1px solid rgba(34,197,94,0.15);
    border-radius: 10px; padding: 12px 16px;
    display: flex; gap: 10px; align-items: flex-start;
    max-width: 820px; margin: 0 auto 4px;
}
.ingest-success-icon { font-size: 15px; margin-top: 1px; flex-shrink: 0; }
.ingest-success-body .title {
    font-size: 12px; font-weight: 500; color: #22C55E;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 3px;
}
.ingest-success-body .meta { font-size: 11px; color: #475569; }

/* scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #161E35; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #1F2D48; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def generate_id() -> str:
    return str(uuid.uuid4())


def add_thread(thread_id: str) -> None:
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat() -> None:
    new_id = generate_id()
    st.session_state["thread_id"] = new_id
    add_thread(new_id)
    st.session_state["message_history"] = []
    st.session_state["pdf_ingested"] = False
    st.session_state["ingest_info"] = None


def load_conversation(thread_id: str):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


# ── session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("message_history", []),
    ("thread_id", generate_id()),
    ("chat_threads", []),
    ("chat_titles", {}),
    ("pdf_ingested", False),
    ("ingest_info", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

add_thread(st.session_state["thread_id"])
tid = st.session_state["thread_id"]


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # brand
    st.markdown("""
    <div class="s-brand">
        <div class="s-brand-icon">🦈</div>
        <div class="s-brand-text">
            <div class="name">SHARKY</div>
            <div class="tag">RAG · SEARCH · TOOLS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # new chat
    st.markdown('<div style="padding:10px 12px 4px">', unsafe_allow_html=True)
    if st.button("＋  New conversation", key="new_chat"):
        reset_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # PDF uploader
    st.markdown('<p class="s-label">Upload document</p>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div style="padding:0 12px">', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "PDF only",
            type=["pdf"],
            label_visibility="collapsed",
            key=f"pdf_upload_{tid}",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded and not st.session_state["pdf_ingested"]:
        with st.spinner("Indexing…"):
            info = ingest_pdf(
                file_bytes=uploaded.read(),
                thread_id=tid,
                filename=uploaded.name,
            )
        st.session_state["pdf_ingested"] = True
        st.session_state["ingest_info"] = info

    # doc badge
    if st.session_state["pdf_ingested"] and st.session_state["ingest_info"]:
        info = st.session_state["ingest_info"]
        st.markdown(f"""
        <div class="doc-badge">
            <div class="doc-badge-icon">📄</div>
            <div class="doc-badge-body">
                <div class="doc-name">{info['filename']}</div>
                <div class="doc-meta">{info['documents']} pages · {info['chunks']} chunks indexed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # conversations
    st.markdown('<p class="s-label" style="margin-top:12px">Conversations</p>', unsafe_allow_html=True)

    threads_rev = st.session_state["chat_threads"][::-1]
    if not threads_rev:
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:10px;color:#1F2D48;padding:6px 16px">No conversations yet</p>', unsafe_allow_html=True)
    else:
        for t_id in threads_rev:
            title = st.session_state["chat_titles"].get(t_id, "New conversation")
            is_active = t_id == st.session_state["thread_id"]
            label = f"{'→ ' if is_active else '   '}{title}"
            if st.button(label, key=f"th_{t_id}"):
                st.session_state["thread_id"] = t_id
                st.session_state["pdf_ingested"] = thread_has_document(t_id)
                meta = thread_document_metadata(t_id)
                st.session_state["ingest_info"] = meta if meta else None
                msgs = load_conversation(t_id)
                temp = []
                for msg in msgs:
                    role = "user" if isinstance(msg, HumanMessage) else "assistant"
                    if isinstance(msg.content, str) and msg.content.strip():
                        temp.append({"role": role, "content": msg.content})
                st.session_state["message_history"] = temp
                st.rerun()

    # tools bar at bottom
    has_doc = st.session_state["pdf_ingested"]
    st.markdown(f"""
    <div class="tools-bar">
        <span class="tool-dot {'active' if has_doc else ''}">RAG</span>
        <span class="tool-dot active">SEARCH</span>
        <span class="tool-dot active">STOCKS</span>
        <span class="tool-dot active">CALC</span>
    </div>
    """, unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
current_title = st.session_state["chat_titles"].get(tid, "New conversation")

st.markdown(f"""
<div class="chat-header">
    <span style="font-size:18px">🦈</span>
    <span class="chat-header-title">{current_title}</span>
    <span class="live-dot">LIVE</span>
</div>
""", unsafe_allow_html=True)

# messages container
msg_container = st.container()

with msg_container:
    if not st.session_state["message_history"]:
        has_doc = st.session_state["pdf_ingested"]
        doc_cap = (
            '<div class="cap-card"><div class="cap-card-icon">✅</div>'
            '<div class="cap-card-title">DOC READY</div>'
            '<div class="cap-card-desc">PDF indexed. Ask anything about it.</div></div>'
            if has_doc else
            '<div class="cap-card"><div class="cap-card-icon">📄</div>'
            '<div class="cap-card-title">PDF RAG</div>'
            '<div class="cap-card-desc">Upload a PDF to ask questions about it.</div></div>'
        )
        st.markdown(f"""
        <div class="empty-state">
            <div class="empty-fin">🦈</div>
            <div class="empty-heading">What do you need?</div>
            <div class="empty-sub">Search the web, crunch numbers, check stocks, or chat with your documents.</div>
            <div class="cap-grid">
                {doc_cap}
                <div class="cap-card">
                    <div class="cap-card-icon">🔍</div>
                    <div class="cap-card-title">WEB SEARCH</div>
                    <div class="cap-card-desc">Real-time results via DuckDuckGo.</div>
                </div>
                <div class="cap-card">
                    <div class="cap-card-icon">📈</div>
                    <div class="cap-card-title">STOCK PRICES</div>
                    <div class="cap-card-desc">Live quotes for any ticker symbol.</div>
                </div>
                <div class="cap-card">
                    <div class="cap-card-icon">🧮</div>
                    <div class="cap-card-title">CALCULATOR</div>
                    <div class="cap-card-desc">Precise arithmetic, no hallucinations.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ingest success banner (shown once after upload)
        if st.session_state["pdf_ingested"] and st.session_state["ingest_info"]:
            info = st.session_state["ingest_info"]
            # only show if last message isn't already about the PDF
            history = st.session_state["message_history"]
            if not history or "indexed" not in (history[-1].get("content", "") or ""):
                st.markdown(f"""
                <div class="ingest-success">
                    <div class="ingest-success-icon">✅</div>
                    <div class="ingest-success-body">
                        <div class="title">{info['filename']} indexed</div>
                        <div class="meta">{info['documents']} pages · {info['chunks']} chunks · RAG ready</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        for message in st.session_state["message_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


# ── INPUT ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything…")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # generate title on first message
    if tid not in st.session_state["chat_titles"]:
        title = generate_chat_title(user_input)
        st.session_state["chat_titles"][tid] = title

    CONFIG = {
        "configurable": {"thread_id": tid},
        "metadata": {"thread_id": tid},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder: dict = {"box": None}

        def ai_only_stream():
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage):
                    name = getattr(chunk, "name", "tool")
                    label = {
                        "duckduckgo_results_json": "searching web",
                        "get_stock_price": f"fetching {name}",
                        "calculator": "computing",
                        "rag_tool": "reading document",
                    }.get(name, f"running {name}")

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"⚙ {label}…", expanded=False)
                    else:
                        status_holder["box"].update(
                            label=f"⚙ {label}…", state="running", expanded=False
                        )

                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(label="Done", state="complete", expanded=False)

    if ai_message:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )