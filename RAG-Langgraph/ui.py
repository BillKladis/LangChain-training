import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import uuid
import json
import os
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from graph import app

# ── Persistence ────────────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")

def load_history() -> dict:
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Deserialise datetime strings back to datetime objects
        for conv in raw.values():
            conv["created_at"] = datetime.fromisoformat(conv["created_at"])
        return raw
    except Exception:
        return {}

def save_history():
    data = {}
    for cid, conv in st.session_state.conversations.items():
        data[cid] = {
            "name": conv["name"],
            "messages": conv["messages"],
            "created_at": conv["created_at"].isoformat(),
        }
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Tighter sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left;
        font-size: 0.875rem;
        padding: 0.4rem 0.75rem;
        border-radius: 6px;
    }
    /* Remove top padding from main area */
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "conversations" not in st.session_state:
    st.session_state.conversations = load_history()

if "active_conv_id" not in st.session_state:
    st.session_state.active_conv_id = None

# ── Helpers ────────────────────────────────────────────────────────────────────
def create_conversation() -> str:
    cid = str(uuid.uuid4())[:8]
    n = len(st.session_state.conversations) + 1
    st.session_state.conversations[cid] = {
        "name": f"New Chat {n}",
        "messages": [],       # {"role": "user"|"assistant", "content": str}
        "created_at": datetime.now(),
    }
    st.session_state.active_conv_id = cid
    return cid


def get_active_conv() -> dict | None:
    return st.session_state.conversations.get(st.session_state.active_conv_id)


def to_lc_messages(msgs: list) -> list:
    """Convert stored Q&A pairs → LangChain message objects.

    Only HumanMessage / AIMessage — no tool calls, no chunks.
    This is what gets injected into the graph as conversation history.
    """
    result = []
    for m in msgs:
        if m["role"] == "user":
            result.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            result.append(AIMessage(content=m["content"]))
    return result

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Conversations")

    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        create_conversation()
        st.rerun()

    st.divider()

    for cid, conv in sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True,
    ):
        is_active = cid == st.session_state.active_conv_id
        prefix = "► " if is_active else "    "
        label = prefix + conv["name"]
        if st.button(label, key=f"btn_{cid}", use_container_width=True):
            st.session_state.active_conv_id = cid
            st.rerun()

# ── Bootstrap: ensure a valid active conversation ──────────────────────────────
if not st.session_state.conversations:
    create_conversation()
    st.rerun()

if st.session_state.active_conv_id not in st.session_state.conversations:
    st.session_state.active_conv_id = next(iter(st.session_state.conversations))
    st.rerun()

# ── Main chat area ─────────────────────────────────────────────────────────────
conv = get_active_conv()

st.title("RAG Research Assistant")
st.caption("Ask anything — I'll search the web and answer from retrieved knowledge.")

# Render existing messages
for msg in conv["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Empty-state prompt
if not conv["messages"]:
    st.markdown(
        "<div style='text-align:center;padding:3rem 0;color:gray;'>"
        "<h3>What would you like to research?</h3>"
        "<p>I'll search the web, extract content, and answer from retrieved knowledge.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Chat input & streaming ─────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything..."):

    # 1. Store and immediately display the user message
    conv["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Build clean history (Q&A only) — exclude the message we just appended
    history = to_lc_messages(conv["messages"][:-1])

    graph_inputs = {
        "messages": history + [HumanMessage(content=prompt)],
        "query": prompt,
        "urls": [],
        "crawled_urls": [],
        "next": "",
        "iterations": 0,
        "rag_completed": False,
    }

    # 3. Stream the graph and render the response progressively
    with st.chat_message("assistant"):
        status_slot = st.empty()   # ephemeral status line
        answer_slot = st.empty()   # grows with each new token
        full_response = ""
        rag_active = False

        try:
            for chunk, metadata in app.stream(graph_inputs, stream_mode="messages"):
                node = metadata.get("langgraph_node", "")

                # Show progress while waiting for the RAG phase
                if not rag_active:
                    if node == "supervisor":
                        status_slot.caption("Planning...")
                    elif node == "research":
                        status_slot.caption("Searching the web and indexing results...")

                # Only render tokens from the RAG node
                if node == "rag":
                    if not rag_active:
                        rag_active = True
                        status_slot.empty()

                    # Whitelist: only plain AI text chunks (no tool calls, no human/tool messages)
                    chunk_type = getattr(chunk, "type", "")
                    if chunk_type not in ("ai", "AIMessageChunk"):
                        continue
                    if getattr(chunk, "tool_calls", None):
                        continue

                    content = getattr(chunk, "content", "")
                    if content:
                        full_response += content
                        # Re-render on every token — Streamlit renders markdown live
                        answer_slot.markdown(full_response + "▌")

            # Final render: remove blinking cursor
            answer_slot.markdown(full_response or "_No answer was generated._")

        except Exception as exc:
            status_slot.empty()
            answer_slot.error(f"Graph execution error: {exc}")

    # 4. Persist the assistant turn (clean text only — no chunks, no tool calls)
    if full_response:
        conv["messages"].append({"role": "assistant", "content": full_response})

        # Auto-name the conversation from its first exchange
        if len(conv["messages"]) == 2:
            conv["name"] = prompt[:35] + ("…" if len(prompt) > 35 else "")

        save_history()

    st.rerun()
