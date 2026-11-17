"""Streamlit chat prototype for the memory-visibility trust study.

Each participant gets a link like:
    http://localhost:8501/?pid=P01&condition=B&session=1

  condition=A  invisible memory: the assistant remembers, the participant sees nothing
  condition=B  visible memory: a side panel shows what's saved, with edit and delete

Conditions are letters, not words, so participants can't tell from the URL.
Open the app with no link parameters to get the researcher setup page.
"""

from contextlib import nullcontext

import streamlit as st

import llm
from memory import MemoryStore

CONDITIONS = {"A": "invisible", "B": "visible"}

SYSTEM_PROMPT = """You are a supportive assistant for a weekly check-in. The user will tell you what is on their mind.

- Be warm, brief (2-4 sentences), and practical. Ask at most one question at a time.
- Do not assume what the user is stressed about. Follow only what they tell you.
- You are not a therapist and do not diagnose. Suggest simple, everyday coping steps.
- You only know what the user has said in this conversation, plus anything listed under "What you remember about the user" below. If that list is absent, you remember nothing about them. Never imply you remember anything else.
- When a listed memory is relevant, bring it up naturally, the way a friend would ("Last time you mentioned...").
- Never invent things the user didn't tell you.
- If the user mentions self-harm, suicide, or being in danger, gently encourage them to call or text 988 (US) or contact emergency services, and keep your reply short and caring."""

GREETING = "Hi! How is your week going? What's on your mind?"


# ---------------------------------------------------------------------------
# Researcher setup page (no link parameters)
# ---------------------------------------------------------------------------
def setup_page():
    st.title("Study setup")
    st.write("Make a participant link. Participants should only ever see the chat page.")
    pid = st.text_input("Participant ID", "P01")
    condition = st.radio(
        "Condition", ["A", "B"],
        format_func=lambda c: f"{c}: {CONDITIONS[c]} memory", horizontal=True,
    )
    session = st.radio("Session", ["1", "2"], horizontal=True)
    st.code(f"http://localhost:8501/?pid={pid}&condition={condition}&session={session}")

    store = MemoryStore(pid)
    st.write(f"Saved memories for {pid}: {len(store.all())}")
    if st.button(f"Reset {pid}'s memories (log is kept)"):
        store.clear()
        st.success("Cleared.")
    st.caption(f"Model: {llm.MODEL} via {llm.BACKEND}")


# ---------------------------------------------------------------------------
# Memory panel (condition B only)
# ---------------------------------------------------------------------------
def _on_edit(store, memory_id, key, session):
    new_text = st.session_state[key].strip()
    if new_text:
        store.edit(memory_id, new_text, session)
    else:  # emptying a memory counts as deleting it
        store.delete(memory_id, session)


def memory_panel(store: MemoryStore, session: str):
    with st.sidebar:
        st.header("What I remember about you")
        st.caption("You can change or delete anything here. I'll only use what's on this list.")
        memories = store.all()
        if not memories:
            st.write("Nothing yet.")
        for m in memories:
            key = f"mem_{m['id']}"
            st.text_area(
                "Memory", m["text"], key=key, label_visibility="collapsed", height=68,
                on_change=_on_edit, args=(store, m["id"], key, session),
            )
            st.button(
                "Delete", key=f"del_{m['id']}",
                on_click=lambda mid=m["id"]: store.delete(mid, session),
            )


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------
def build_messages(store: MemoryStore, history):
    memories = store.for_prompt()
    system = SYSTEM_PROMPT
    if memories:
        system += "\n\nWhat you remember about the user:\n" + "\n".join(f"- {m['text']}" for m in memories)
    return [{"role": "system", "content": system}] + history, [m["id"] for m in memories]


def chat_page(pid: str, condition: str, session: str):
    store = MemoryStore(pid)
    visible = CONDITIONS[condition] == "visible"

    if "history" not in st.session_state:
        st.session_state.history = []  # chat starts fresh each session; memory carries over
        store.log("session_started", session, condition=condition)

    if visible:
        memory_panel(store, session)
        for text in st.session_state.pop("just_saved", []):
            st.toast(f"Saved to memory: {text}")

    st.title("Weekly check-in")
    st.caption("Research prototype. Please use the scenario you were given, not your own personal details.")

    with st.chat_message("assistant"):
        st.write(GREETING)
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_text = st.chat_input("Type a message")
    if not user_text:
        return

    with st.chat_message("user"):
        st.write(user_text)
    st.session_state.history.append({"role": "user", "content": user_text})
    store.log("user_message", session, text=user_text)

    messages, used_ids = build_messages(store, st.session_state.history)
    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(llm.chat_stream(messages))
        except Exception as e:  # most often: Ollama isn't running
            st.error(f"Couldn't reach the model ({e}). Is Ollama running? Try `ollama serve`.")
            st.session_state.history.pop()
            return
    st.session_state.history.append({"role": "assistant", "content": reply})
    # memories_used lets you check recall accuracy later (did the model use what was saved?)
    store.log("assistant_message", session, text=reply, memories_used=used_ids)

    with st.spinner("Updating memory...") if visible else nullcontext():
        saved = store.extract_and_save(user_text, session)
    if visible and saved:
        st.session_state.just_saved = [m["text"] for m in saved]
        st.rerun()  # refresh the panel so new memories show up


# ---------------------------------------------------------------------------
st.set_page_config(page_title="Weekly check-in", page_icon="💬")
params = st.query_params
pid = params.get("pid")
condition = params.get("condition", "").upper()
session = params.get("session", "1")

if pid and condition in CONDITIONS:
    chat_page(pid, condition, session)
else:
    setup_page()
