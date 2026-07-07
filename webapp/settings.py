"""Web app configuration, all environment-driven (12-factor style).

Local dev reads .env via python-dotenv; on Railway the same variables are
set in the service's environment. Nothing here is ever rendered in a page
or written to logs.
"""
from __future__ import annotations

import os
import secrets

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# --- auth ---
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "")  # bcrypt; scripts/hash_password.py

# Session-cookie signing key. A missing SECRET_KEY gets a random per-process
# key so local dev still works, but every restart logs everyone out — set it.
SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
SECRET_KEY_WAS_GENERATED = "SECRET_KEY" not in os.environ

SESSION_COOKIE = "session"
SESSION_MAX_AGE_SECONDS = 7 * 24 * 3600

# Login rate limit: max failures per IP within the window, then 429.
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SECONDS = 900

# --- database ---
# SQLite locally; Railway injects a postgres:// DATABASE_URL (normalized in db.py).
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./stock_evaluator.db")

# --- analysis behavior ---
ANALYSIS_CACHE_HOURS = _int_env("ANALYSIS_CACHE_HOURS", 24)
MAX_CONCURRENT_ANALYSES = _int_env("MAX_CONCURRENT_ANALYSES", 2)
