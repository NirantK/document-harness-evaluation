"""Persistent cache of model replies, keyed by the exact request.

Temperature 0 and a fixed seed make a request repeatable, so an identical request need not reach the model again. A question
replays instantly up to the first turn whose prompt changed, which is what makes re-running a revision or resuming a stopped
run cheap. An LRU in memory sits in front of SQLite.
"""

import hashlib
import json
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parents[3] / ".state" / "talk2_response_cache.sqlite3"
MEMORY_ENTRIES = 4096
_lock = threading.Lock()
_connection: sqlite3.Connection | None = None


def connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _connection = sqlite3.connect(CACHE_PATH, check_same_thread=False, timeout=30)
        _connection.execute("PRAGMA journal_mode=WAL")
        _connection.execute("CREATE TABLE IF NOT EXISTS reply (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return _connection


def request_key(request: dict) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=MEMORY_ENTRIES)
def stored_reply(key: str) -> str:
    """The stored reply. A miss raises KeyError, which lru_cache does not remember, so a later store is seen."""
    with _lock:
        row = connection().execute("SELECT value FROM reply WHERE key = ?", (key,)).fetchone()
    if row is None:
        raise KeyError(key)
    return row[0]


def store_reply(key: str, value: dict) -> None:
    with _lock:
        connection().execute("INSERT OR REPLACE INTO reply VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))
        connection().commit()
