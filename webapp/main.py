"""FastAPI app: a thin, authenticated web wrapper around the analysis engine.

Run locally:   uvicorn webapp.main:app --reload
On Railway:    uvicorn webapp.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from webapp import auth, service, settings
from webapp.db import init_db

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("webapp")

app = FastAPI(title="Stock Evaluator", docs_url=None, redoc_url=None, openapi_url=None)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _fmt_money(v, currency: str = "") -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 1e9:
        return f"{v / 1e9:.2f}B {currency}".strip()
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M {currency}".strip()
    return f"{v:,.2f} {currency}".strip()


def _fmt_pct0(v) -> str:
    return f"{v:.0%}" if v is not None else "n/a"


def _fmt_pct1(v) -> str:
    return f"{v:.1%}" if v is not None else "n/a"


def _fmt_ratio(v, kind: str) -> str:
    if v is None:
        return "n/a"
    if kind == "pct":
        return f"{v:.1%}"
    if kind == "x":
        return f"{v:.2f}x"
    if kind == "money":
        return _fmt_money(v)
    return f"{v:.2f}"


templates.env.filters["money"] = _fmt_money
templates.env.filters["pct0"] = _fmt_pct0
templates.env.filters["pct1"] = _fmt_pct1
templates.env.filters["rfmt"] = _fmt_ratio

# Same grouping as the CLI's Key Ratios table.
RATIO_GROUPS = [
    ("Profitability", [
        ("gross_margin", "Gross margin", "pct"), ("operating_margin", "Operating margin", "pct"),
        ("net_margin", "Net margin", "pct"), ("ebitda_margin", "EBITDA margin", "pct"),
        ("roe", "ROE", "pct"), ("roa", "ROA", "pct"), ("roic", "ROIC", "pct"),
    ]),
    ("Growth", [
        ("revenue_growth_yoy", "Revenue growth (YoY)", "pct"), ("revenue_cagr_3y", "Revenue CAGR (3y)", "pct"),
        ("eps_growth_yoy", "EPS growth (YoY)", "pct"), ("fcf_growth_yoy", "FCF growth (YoY)", "pct"),
    ]),
    ("Leverage / Liquidity", [
        ("debt_to_equity", "Debt / Equity", "num"), ("net_debt_to_ebitda", "Net debt / EBITDA", "x"),
        ("current_ratio", "Current ratio", "num"), ("quick_ratio", "Quick ratio", "num"),
        ("interest_coverage", "Interest coverage", "x"),
    ]),
    ("Cash Flow", [
        ("free_cash_flow", "Free cash flow", "money"), ("fcf_margin", "FCF margin", "pct"),
        ("fcf_yield", "FCF yield", "pct"), ("capex_to_revenue", "Capex / Revenue", "pct"),
    ]),
    ("Valuation", [
        ("pe_trailing", "P/E (trailing)", "num"), ("pe_forward", "P/E (forward)", "num"),
        ("price_to_book", "Price / Book", "num"), ("price_to_sales", "Price / Sales", "num"),
        ("ev_to_ebitda", "EV / EBITDA", "num"), ("ev_to_revenue", "EV / Revenue", "num"),
        ("peg_ratio", "PEG ratio", "num"),
    ]),
    ("Dividend", [
        ("dividend_yield", "Dividend yield", "pct"), ("payout_ratio", "Payout ratio", "pct"),
    ]),
]

TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,10}$")

# Everything is behind auth except these exact paths.
PUBLIC_PATHS = {"/login", "/health"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in PUBLIC_PATHS or auth.session_is_valid(request):
            return await call_next(request)
        if request.method == "GET":
            return RedirectResponse("/login", status_code=303)
        return JSONResponse({"error": "authentication required"}, status_code=401)


app.add_middleware(AuthMiddleware)


@app.on_event("startup")
def startup() -> None:
    init_db()
    if settings.SECRET_KEY_WAS_GENERATED:
        log.warning("SECRET_KEY not set — using a random per-process key; sessions reset on restart.")
    if not settings.ADMIN_PASSWORD_HASH:
        log.warning("ADMIN_PASSWORD_HASH not set — login is impossible until it is configured.")


def _set_session(response, request: Request) -> None:
    response.set_cookie(
        settings.SESSION_COOKIE,
        auth.make_session_cookie(),
        max_age=settings.SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
    )


# ---------------- public routes ----------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if auth.session_is_valid(request):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    ip = auth.client_ip(request)
    if auth.login_rate_limited(ip):
        return templates.TemplateResponse(
            request, "login.html",
            {"error": "Too many failed attempts — try again in a few minutes."},
            status_code=429,
        )
    if not auth.verify_credentials(username, password):
        auth.record_login_failure(ip)
        return templates.TemplateResponse(
            request, "login.html", {"error": "Invalid username or password."}, status_code=401
        )
    auth.clear_login_failures(ip)
    response = RedirectResponse("/", status_code=303)
    _set_session(response, request)
    return response


# ---------------- authenticated routes ----------------

@app.post("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(settings.SESSION_COOKIE)
    return response


@app.get("/", response_class=HTMLResponse)
def index(request: Request, msg: str = ""):
    return templates.TemplateResponse(
        request, "index.html",
        {"analyses": service.recent_analyses(), "msg": msg},
    )


@app.post("/analyze")
def analyze(request: Request, background: BackgroundTasks, ticker: str = Form(...), force: str = Form("")):
    ticker = ticker.strip().upper()
    if not TICKER_RE.match(ticker):
        return RedirectResponse("/?msg=Invalid+ticker", status_code=303)

    # Duplicate guard: an analysis for this ticker is already in flight.
    active = service.find_active(ticker)
    if active is not None:
        return RedirectResponse(f"/analysis/{active.id}", status_code=303)

    # Cache: recent completed analysis -> serve it instead of spending money.
    if not force:
        cached = service.find_cached(ticker)
        if cached is not None:
            return RedirectResponse(f"/analysis/{cached.id}?cached=1", status_code=303)

    # Concurrency guard: each run costs real API money.
    if service.active_count() >= settings.MAX_CONCURRENT_ANALYSES:
        return RedirectResponse(
            f"/?msg=Busy:+{settings.MAX_CONCURRENT_ANALYSES}+analyses+already+running,+try+again+shortly",
            status_code=303,
        )

    analysis = service.create_analysis(ticker)
    background.add_task(service.run_analysis_task, analysis.id)
    return RedirectResponse(f"/analysis/{analysis.id}", status_code=303)


@app.get("/analysis/{analysis_id}", response_class=HTMLResponse)
def analysis_page(request: Request, analysis_id: int, cached: str = ""):
    analysis = service.get_analysis(analysis_id)
    if analysis is None:
        return RedirectResponse("/?msg=Analysis+not+found", status_code=303)
    return templates.TemplateResponse(
        request, "analysis.html",
        {"a": analysis, "payload": analysis.payload, "from_cache": bool(cached), "ratio_groups": RATIO_GROUPS},
    )


@app.get("/analysis/{analysis_id}/status")
def analysis_status(analysis_id: int):
    analysis = service.get_analysis(analysis_id)
    if analysis is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"id": analysis.id, "status": analysis.status, "error": analysis.error}
