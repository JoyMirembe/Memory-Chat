# Memory chat prototype (Continual Learning in AI final project)

A Streamlit chat app for our study on whether seeing (and controlling) what an AI remembers changes trust.
It uses one model and one memory layer. The memory panel is the only thing that differs between conditions.

- **Condition A (invisible memory):** the assistant saves and uses memories, but the participant sees nothing.
- **Condition B (visible memory):** a side panel called "What I remember about you" lists every saved memory with edit and delete.

## Setup (once per laptop)

1. Install Ollama from https://ollama.com and download the model:
   ```
   ollama pull qwen3:8b
   ```
   On a weaker laptop, use `qwen3:4b` and set `LLM_MODEL=qwen3:4b` before running.
2. Install the Python packages:
   ```
   pip install -r requirements.txt
   ```

## Running a session

```
streamlit run app.py
```

Open http://localhost:8501 with nothing added to the address to get the **researcher setup page**. Pick the participant ID, condition and session, and it gives you the participant's link, for example:

```
http://localhost:8501/?pid=P01&condition=B&session=1
```

- Use the **same pid and condition** for session 1 and session 2. The chat starts fresh each session, but memories carry over, so in session 2 the assistant can bring up what was said in session 1.
- Conditions are letters (A/B) so participants can't tell from the link.
- To test the interface without Ollama, run `LLM_BACKEND=mock streamlit run app.py` (it gives fake replies).

## What gets recorded

Everything is saved in the `data/` folder, one file per participant:

- `data/memories/P01.json` holds the participant's current memories
- `data/logs/P01.jsonl` has one line per event: `session_started`, `user_message`, `assistant_message`, `memory_saved`, `memory_edited`, `memory_deleted`

Useful for analysis:
- **Behavioral measure:** count `memory_edited` and `memory_deleted` events for condition B participants.
- **Recall check (our confound check):** each `assistant_message` lists `memories_used`, the memories the model was given that turn. Compare that against what it actually said to spot moments when the AI remembered wrong.

## How the memory works (plain version)

1. After each participant message, the model is asked a second, hidden question: "what in this message is worth remembering?" It returns short facts like "Has a statistics exam on Friday."
2. Those facts are saved to the participant's file.
3. On every turn, all saved facts (up to 30) are added to the model's instructions, so it can bring them up naturally.

We put every memory in the prompt instead of searching for the most relevant ones. Study sessions are short, so there are only a handful of facts, and skipping the search step means fewer chances for the model to forget something. That keeps memory errors from muddying the trust comparison.

## Files

- `app.py`: the chat page, memory panel, and researcher setup page
- `memory.py`: saving, editing and deleting memories, the extraction prompt, and logging
- `llm.py`: talks to Ollama (or the mock)
