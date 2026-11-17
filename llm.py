"""Talks to the language model.

Default: Qwen3 4B Instruct (2507, non-reasoning) running locally through Ollama (http://localhost:11434).
Set LLM_BACKEND=mock to run the app without a model (for testing the interface).
"""

import json
import os
import re

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("LLM_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
BACKEND = os.environ.get("LLM_BACKEND", "ollama")

# Speed settings (same for every participant and condition)
KEEP_ALIVE = "30m"   # keep the model loaded between turns (default is 5 min -> slow reloads)
NUM_CTX = 2048       # small context window: less memory use, faster on an 8 GB laptop
REPLY_TOKENS = 200   # replies are 2-4 sentences, so cap the length
EXTRACT_TOKENS = 150 # memory facts are short JSON

THINKING_MODEL = "qwen3" in MODEL.lower() and "instruct" not in MODEL.lower()  # only reasoning models need the buffered path

THINK_TAGS = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean(text: str) -> str:
    """Remove any <think>...</think> reasoning Qwen3 might still emit."""
    text = THINK_TAGS.sub("", text)
    if "</think>" in text:  # opening tag missing: drop everything up to the close
        text = text.split("</think>", 1)[1]
    return text.strip()


def _prep(messages):
    """Copy the messages and add Qwen3's /no_think switch to the system prompt
    (belt and braces: some Ollama versions ignore think=False)."""
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] += "\n\n/no_think"
    return msgs


def chat_stream(messages):
    """Yield the assistant's reply piece by piece (for a typing effect).

    The full reply is collected first and cleaned, THEN replayed word by word.
    Streaming raw chunks can't be made safe: Qwen3 sometimes emits its reasoning
    with no opening <think> tag, so the only reliable signal is the closing
    </think>, which arrives after the reasoning has already been shown.
    The typing effect is identical for every participant and both conditions.
    """
    if BACKEND == "mock":
        yield from _mock_reply(messages)
        return

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": _prep(messages),
            "stream": True,
            "think": False,  # no visible reasoning in the chat
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": 0.7, "num_ctx": NUM_CTX, "num_predict": REPLY_TOKENS},
        },
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()
    if not THINKING_MODEL:
        # Non-reasoning model: nothing to hide, so stream live (fastest to first word)
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line).get("message", {}).get("content", "")
                if chunk:
                    yield chunk
        return
    parts = []
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line).get("message", {}).get("content", "")
        if chunk:
            parts.append(chunk)
    text = _clean("".join(parts))
    for word in text.split(" "):
        yield word + " "


def chat_json(messages) -> dict:
    """Ask the model for a JSON answer (used for memory extraction)."""
    if BACKEND == "mock":
        return _mock_extract(messages)

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": _prep(messages),
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": EXTRACT_TOKENS},  # consistent extraction
        },
        timeout=300,
    )
    resp.raise_for_status()
    text = _clean(resp.json()["message"]["content"])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ---------------- mock backend (no model needed) ----------------

def _mock_reply(messages):
    system = messages[0]["content"]
    user = messages[-1]["content"]
    remembered = "nothing yet"
    if "What you remember about the user:" in system:
        remembered = system.split("What you remember about the user:")[1].strip().splitlines()[0]
    text = (
        f"[mock reply] That sounds like a lot. You said: \"{user[:80]}\". "
        f"(First memory in context: {remembered})"
    )
    for word in text.split(" "):
        yield word + " "


def _mock_extract(messages):
    user = messages[-1]["content"]
    # Pretend anything after "Message:" containing "I " is a fact worth saving
    msg = user.split("Message:", 1)[-1].strip()
    if len(msg.split()) >= 5:
        return {"memories": [f"User said: {msg[:100]}"]}
    return {"memories": []}
