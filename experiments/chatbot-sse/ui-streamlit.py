import json
import os
import uuid
from datetime import datetime

import httpx
import streamlit as st
from httpx_sse import connect_sse

HISTORY_DIR = "chat_logs"
SERVER_URL = "http://localhost:8000/chat"
TITLE_URL = "http://localhost:8000/title"
SYSTEM_PROMPT = "You are a witty, helpful assistant. Keep your answer brief, preferably less than 3 sentences, unless asked for details."


def make_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{timestamp}-{suffix}"


def save_history(session_id: str, metadata: dict, history: list[dict]):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    payload = {"metadata": metadata, "messages": history}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_history(session_id: str) -> tuple[dict, list[dict]]:
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        # Legacy shape: file IS the message list, no metadata wrapper yet.
        return {"title": None}, data
    return data.get("metadata", {"title": None}), data.get("messages", [])


def generate_title(history: list[dict]) -> str:
    """Calls the server's /title endpoint to summarize the conversation so far."""
    convo = [m for m in history if m["role"] != "system"]
    response = httpx.post(TITLE_URL, json={"messages": convo}, timeout=30.0)
    response.raise_for_status()
    return response.json()["title"]


@st.cache_data
def _title_from_file(path: str, mtime: float) -> str:
    """
    mtime is part of the cache key purely to invalidate when the file
    changes. Prefers an LLM-generated metadata.title; falls back to a
    truncated first-user-message heuristic for legacy files or sessions
    where titling hasn't run yet (e.g. the failed-mid-stream case).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        messages, title = data, None
    else:
        messages = data.get("messages", [])
        title = data.get("metadata", {}).get("title")
    if title:
        return title
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"].strip().replace("\n", " ")
            return text[:40] + ("…" if len(text) > 40 else "")
    return "New chat"


def list_sessions() -> list[tuple[str, str]]:
    """Returns (session_id, title) pairs, most recently modified first."""
    if not os.path.isdir(HISTORY_DIR):
        return []
    entries = []
    for fname in os.listdir(HISTORY_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(HISTORY_DIR, fname)
        mtime = os.path.getmtime(path)
        session_id = fname[: -len(".json")]
        entries.append((session_id, _title_from_file(path, mtime), mtime))
    entries.sort(key=lambda e: e[2], reverse=True)
    return [(session_id, title) for session_id, title, _ in entries]


def stream_chat_sync(history: list[dict]):
    """
    Sync generator version of client.py's stream_chat, for st.write_stream.
    Yields raw token strings; raises RuntimeError on a server-side error event.
    """
    with httpx.Client(timeout=60.0) as http_client:
        with connect_sse(
            http_client,
            "POST",
            SERVER_URL,
            json={"messages": history},
        ) as event_source:
            for sse in event_source.iter_sse():
                if sse.event == "token":
                    yield json.loads(sse.data)["token"]
                elif sse.event == "done":
                    break
                elif sse.event == "error":
                    message = json.loads(sse.data).get("message", "unknown server error") if sse.data else "unknown server error"
                    raise RuntimeError(f"Server error mid-stream: {message}")


# ---------------------------------------------------------------------------
# Session state init (persists across reruns within a browser session)
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = make_session_id()
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]
if "metadata" not in st.session_state:
    st.session_state.metadata = {"title": None}

st.set_page_config(page_title="GPT-4o Chat", page_icon="💬")

with st.sidebar:
    st.header("Chats")
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.session_id = make_session_id()
        st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.metadata = {"title": None}
        st.rerun()

    st.divider()
    for session_id, title in list_sessions():
        is_active = session_id == st.session_state.session_id
        if st.button(
            title,
            key=f"sess_{session_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.session_id = session_id
            st.session_state.metadata, st.session_state.history = load_history(session_id)
            st.rerun()

st.title("💬 Chat")

# Render existing history (skip system prompt)
for msg in st.session_state.history:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            full_response = st.write_stream(stream_chat_sync(st.session_state.history))
        except RuntimeError as e:
            st.error(str(e))
            # mirror client.py: drop the unanswered user turn so history stays consistent
            st.session_state.history.pop()
            full_response = None

    if full_response is not None:
        st.session_state.history.append({"role": "assistant", "content": full_response})

        # First full exchange = system + user + assistant = 3 messages.
        # Title is generated exactly once and never re-evaluated after.
        if st.session_state.metadata.get("title") is None and len(st.session_state.history) == 3:
            try:
                st.session_state.metadata["title"] = generate_title(st.session_state.history)
            except (httpx.HTTPError, KeyError):
                pass  # sidebar falls back to the truncated-first-message heuristic

        save_history(st.session_state.session_id, st.session_state.metadata, st.session_state.history)