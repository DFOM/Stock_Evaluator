"""Single-admin session auth.

- Credentials: ADMIN_USERNAME + bcrypt ADMIN_PASSWORD_HASH from env
  (generate the hash with scripts/hash_password.py).
- Session: signed, tamper-proof cookie (itsdangerous + SECRET_KEY),
  HttpOnly + SameSite=strict. SameSite=strict is the CSRF story for this
  single-admin app: no cross-site request can carry the session cookie.
- Login rate limit: in-memory failure tracking per client IP (fine for a
  single-process deployment, which this is).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

import bcrypt
from fastapi import Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from webapp import settings

_serializer = URLSafeTimedSerializer(settings.SECRET_KEY, salt="session")

_failures: dict[str, deque[float]] = defaultdict(deque)


def verify_credentials(username: str, password: str) -> bool:
    if not settings.ADMIN_PASSWORD_HASH:
        return False
    if username != settings.ADMIN_USERNAME:
        # Still burn a bcrypt comparison so username validity isn't timeable.
        bcrypt.checkpw(b"x", b"$2b$12$" + b"0" * 53)
        return False
    try:
        return bcrypt.checkpw(password.encode(), settings.ADMIN_PASSWORD_HASH.encode())
    except ValueError:
        return False  # malformed hash in env


def make_session_cookie() -> str:
    return _serializer.dumps({"u": settings.ADMIN_USERNAME})


def session_is_valid(request: Request) -> bool:
    raw = request.cookies.get(settings.SESSION_COOKIE)
    if not raw:
        return False
    try:
        data = _serializer.loads(raw, max_age=settings.SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return False
    return data.get("u") == settings.ADMIN_USERNAME


def client_ip(request: Request) -> str:
    # Railway/most proxies set X-Forwarded-For; leftmost is the client.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def login_rate_limited(ip: str) -> bool:
    q = _failures[ip]
    cutoff = time.time() - settings.LOGIN_WINDOW_SECONDS
    while q and q[0] < cutoff:
        q.popleft()
    return len(q) >= settings.LOGIN_MAX_FAILURES


def record_login_failure(ip: str) -> None:
    _failures[ip].append(time.time())


def clear_login_failures(ip: str) -> None:
    _failures.pop(ip, None)
