"""Chat history persistence."""

import json
import os

CHAT_HISTORY_PATH = "data/chat_history.json"


def load_history() -> list:
    try:
        if os.path.exists(CHAT_HISTORY_PATH):
            with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_history(messages: list) -> None:
    os.makedirs("data", exist_ok=True)
    try:
        with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
