"""
Streamlit UI for the Mental Health RAG chatbot — dark theme.

Run with:
    streamlit run streamlit_app.py

Assumes src/app.py (your pasted script) sits at Mental-Health-Assitant/src/app.py,
and that PINECONE_API_KEY, PINECONE_CLOUD, PINECONE_REGION, and
GOOGLE_API_KEY are set in a .env file or the environment — same as
your original CLI script expected.
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Heavy imports (Pinecone client, embeddings model, LLM, chain construction)
# all happen at import time inside src/app.py, exactly like your
# original script — so importing this module IS the "startup cost."
#
# This import assumes you run `streamlit run streamlit_app.py` from the
# repo root (Mental-Health-Assitant/), with src/ as a subfolder next to
# this file. Python 3 treats src/ as an implicit namespace package, so
# no src/__init__.py is required.
from src import app as backend


st.set_page_config(
    page_title="Steady — Mental Health RAG Assistant",
    page_icon="◐",
    layout="centered",
)

# ---------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------
# Palette (named, deliberate — not the generic near-black+acid-green combo):
#   ink        #14171A  page background, warm-toned near-black, not pure #000
#   surface    #1C2023  card / bubble background
#   surface-2  #23282B  secondary surface (sidebar, source panel)
#   line       #2E3438  hairline borders
#   text       #E9E6E0  primary text, warm off-white (not stark #FFFFFF)
#   muted      #9B9791  secondary text
#   sage       #7FA187  accent — calm, grounded, not alarm-colored
#   clay       #C98B5D  secondary accent, used sparingly (user bubble edge)
#
# Type:
#   display -> "Fraunces"     (warm humanist serif, used only for the title)
#   body    -> "Inter"        (clean, high-legibility sans)
#   mono    -> "JetBrains Mono" (source/doc-id chips)
#
# Note: Streamlit's internal CSS classes and data-testid attributes are
# undocumented and can change between Streamlit versions. This styling
# was written against a recent-ish Streamlit release from memory — if a
# section doesn't visually apply after you upgrade/downgrade Streamlit,
# open devtools, find the new attribute name, and swap it in below.

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --ink: #14171A;
        --surface: #1C2023;
        --surface-2: #23282B;
        --line: #2E3438;
        --text: #E9E6E0;
        --muted: #9B9791;
        --sage: #7FA187;
        --clay: #C98B5D;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background-color: var(--surface-2);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    /* Title block */
    .steady-title {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.1rem;
        color: var(--text);
        letter-spacing: -0.01em;
        margin-bottom: 0.15rem;
    }
    .steady-accent {
        color: var(--sage);
    }
    .steady-rule {
        height: 1px;
        background: linear-gradient(90deg, var(--sage) 0%, var(--line) 40%);
        border: none;
        margin: 0.6rem 0 1.4rem 0;
    }
    .steady-caption {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background-color: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.4rem 0.2rem;
        margin-bottom: 0.9rem;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: var(--text) !important;
    }
    /* user turns get a thin clay left-edge, assistant turns a sage one */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        border-left: 2px solid var(--clay);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        border-left: 2px solid var(--sage);
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background-color: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--surface-2);
        color: var(--text);
        border: 1px solid var(--line);
        border-radius: 10px;
        transition: border-color 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--sage);
        color: var(--sage);
    }

    /* Expander (sources panel) */
    [data-testid="stExpander"] {
        background-color: var(--surface-2);
        border: 1px solid var(--line);
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary {
        color: var(--muted) !important;
        font-size: 0.85rem;
    }
    [data-testid="stExpander"] p, [data-testid="stExpander"] div {
        color: var(--text) !important;
    }
    .source-chip {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--sage);
        background-color: rgba(127, 161, 135, 0.12);
        border: 1px solid rgba(127, 161, 135, 0.35);
        border-radius: 6px;
        padding: 1px 8px;
        margin-bottom: 4px;
    }

    /* Focus visibility (accessibility floor — not skipped for the theme) */
    *:focus-visible {
        outline: 2px solid var(--sage) !important;
        outline-offset: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    '<div class="steady-title">Steady <span class="steady-accent">◐</span></div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="steady-rule">', unsafe_allow_html=True)
st.markdown(
    '<div class="steady-caption">Answers are grounded only in your Pinecone-indexed '
    "knowledge base (index: <code>medical-chatbot</code>). Not a substitute for "
    "professional care.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "display_history" not in st.session_state:
    st.session_state.display_history = []  # list[dict]
if "lc_history" not in st.session_state:
    st.session_state.lc_history = []  # list[HumanMessage | AIMessage]

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("**Session**")
    st.write(f"Turns so far: **{len(st.session_state.lc_history) // 2}**")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.display_history = []
        st.session_state.lc_history = []
        st.rerun()

    st.divider()
    st.caption(
        "Streams from `rag_chain.stream(...)` in src/app.py. If retrieval "
        "or the LLM call errors (bad API key, missing Pinecone index, etc.), "
        "the error shows inline in the chat — check your `.env` values first."
    )

# ---------------------------------------------------------------------
# Render existing conversation
# ---------------------------------------------------------------------
for turn in st.session_state.display_history:
    avatar = "🧑" if turn["role"] == "user" else "◐"
    with st.chat_message(turn["role"], avatar=avatar):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for i, doc in enumerate(turn["sources"], start=1):
                    st.markdown(
                        f'<span class="source-chip">{doc["doc_id"]}</span>',
                        unsafe_allow_html=True,
                    )
                    preview = doc.get("text", "")
                    st.text(preview[:500] + ("..." if len(preview) > 500 else ""))

# ---------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------
user_question = st.chat_input("Ask a question related to mental health...")

if user_question:
    st.session_state.display_history.append({"role": "user", "content": user_question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar="◐"):
        placeholder = st.empty()
        full_text = ""
        sources_raw = []
        error = None

        try:
            for chunk in backend.rag_chain.stream(
                {
                    "input": user_question,
                    "chat_history": st.session_state.lc_history,
                }
            ):
                if "context" in chunk and chunk["context"]:
                    sources_raw = chunk["context"]
                if "answer" in chunk and chunk["answer"] is not None:
                    full_text += chunk["answer"]
                    placeholder.markdown(full_text + "▌")

            placeholder.markdown(full_text if full_text else "_(no answer returned)_")

        except Exception as e:
            error = str(e)
            placeholder.error(
                "Something went wrong calling the RAG chain. This usually means "
                "a missing/invalid API key, an unreachable Pinecone index, or a "
                "LangChain/library version mismatch — verify your `.env` and "
                f"installed package versions.\n\nRaw error: `{error}`"
            )

        if error is None:
            st.session_state.lc_history.append(HumanMessage(content=user_question))
            st.session_state.lc_history.append(AIMessage(content=full_text))

            sources_for_display = [
                {
                    "doc_id": backend.get_document_id(d),
                    "text": d.page_content,
                    "metadata": d.metadata or {},
                }
                for d in sources_raw
            ]

            st.session_state.display_history.append(
                {
                    "role": "assistant",
                    "content": full_text,
                    "sources": sources_for_display,
                }
            )

            if sources_for_display:
                with st.expander(f"Sources ({len(sources_for_display)})"):
                    for i, doc in enumerate(sources_for_display, start=1):
                        st.markdown(
                            f'<span class="source-chip">{doc["doc_id"]}</span>',
                            unsafe_allow_html=True,
                        )
                        preview = doc["text"]
                        st.text(preview[:500] + ("..." if len(preview) > 500 else ""))
