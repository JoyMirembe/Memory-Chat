"""The memory layer, plus study logging.

Both conditions use exactly the same memory layer. The only difference
between conditions is whether the participant can SEE and change it (app.py).

Files, one per participant:
  data/memories/<pid>.json   saved memories (carried from session 1 to session 2)
  data/logs/<pid>.jsonl      every event: messages, memories saved/edited/deleted,
                             and which memories the model was given each turn
"""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import llm

DATA_DIR = Path(__file__).parent / "data"
MAX_MEMORIES_IN_PROMPT = 30

EXTRACTION_PROMPT = """You decide what a supportive assistant should remember long-term about a user.

Read ONLY the user's latest message and pull out personal facts the user explicitly stated in it:
their situation, stressors, deadlines, people in their life, preferences, and coping strategies.

Rules:
- Write each fact as one short sentence in the third person, e.g. "Is worried about [thing the user mentioned]."
- The example above is only a format guide. Never copy wording from these instructions, and never output square brackets.
- Only include facts the user clearly said. Do not guess, interpret, or fill in likely details.
- Greetings, small talk, thanks, and requests for help with no personal details ("hi", "can you help me?") contain nothing to remember: return {"memories": []}.
- Skip anything already in the existing memories.
- If there is nothing new, return {"memories": []}.

Return JSON only, in this format: {"memories": ["fact 1", "fact 2"]}"""


def _supported(fact: str, message: str) -> bool:
    """Safeguard against phantom memories: the fact must share at least one
    content word (matched on the first 4 letters, so exam/exams match) with the user's message."""
    def stems(t):
        return {w[:4] for w in re.findall(r"[a-z0-9']+", t.lower()) if len(w) >= 4}
    return bool(stems(fact) & stems(message))


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, pid: str):
        self.pid = pid
        self.mem_path = DATA_DIR / "memories" / f"{pid}.json"
        self.log_path = DATA_DIR / "logs" / f"{pid}.jsonl"
        self.mem_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- reading / writing ----------
    def all(self):
        if not self.mem_path.exists():
            return []
        return json.loads(self.mem_path.read_text())

    def _save(self, memories):
        self.mem_path.write_text(json.dumps(memories, indent=2))

    def add(self, text, session):
        memories = self.all()
        item = {"id": uuid.uuid4().hex[:8], "text": text, "created": _now(), "session": session}
        memories.append(item)
        self._save(memories)
        self.log("memory_saved", session, memory_id=item["id"], text=text)
        return item

    def edit(self, memory_id, new_text, session):
        memories = self.all()
        for m in memories:
            if m["id"] == memory_id and m["text"] != new_text:
                self.log("memory_edited", session, memory_id=memory_id, old_text=m["text"], new_text=new_text)
                m["text"] = new_text
        self._save(memories)

    def delete(self, memory_id, session):
        memories = self.all()
        gone = [m for m in memories if m["id"] == memory_id]
        self._save([m for m in memories if m["id"] != memory_id])
        if gone:
            self.log("memory_deleted", session, memory_id=memory_id, text=gone[0]["text"])

    def clear(self):
        """Researcher reset. Deletes this participant's memories (log is kept)."""
        if self.mem_path.exists():
            self.mem_path.unlink()

    # ---------- using memory ----------
    def for_prompt(self):
        """The memories handed to the model this turn (most recent first if over the cap).

        With a short study, every memory fits in the prompt, so there is no
        search step that could miss a fact. That keeps 'the model forgot'
        errors to a minimum, which matters for the trust comparison.
        """
        return self.all()[-MAX_MEMORIES_IN_PROMPT:]

    def extract_and_save(self, user_message, session):
        """Ask the model what's worth remembering from the latest message, then save it."""
        existing = "\n".join(f"- {m['text']}" for m in self.all()) or "(none)"
        result = llm.chat_json([
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": f"Existing memories:\n{existing}\n\nMessage: {user_message}"},
        ])
        facts = result.get("memories", []) if isinstance(result, dict) else []
        saved = []
        for fact in facts:
            if not (isinstance(fact, str) and fact.strip()):
                continue
            fact = fact.strip()
            if "[" in fact or not _supported(fact, user_message):
                # logged so the phantom-memory rate can be reported in the write-up
                self.log("memory_rejected", session, text=fact, user_message=user_message)
                continue
            saved.append(self.add(fact, session))
        return saved

    # ---------- study log ----------
    def log(self, event, session, **fields):
        record = {"time": _now(), "pid": self.pid, "session": session, "event": event, **fields}
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
