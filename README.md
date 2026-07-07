# Stock Evaluator

Personal stock evaluation tool: enter a ticker, get a full analysis card.

A rule-based engine (ratios → business-shape archetype → pillar scores →
red-flag vetoes → archetype-specific fair value) is combined with a
**dual-pass AI analyst ensemble**: pluggable LLM analysts first value the
company **price-blind** (their ranges form the fair-value consensus), then
see the full packet as **critics** (risks, archetype disputes, method
critiques — their sighted fair values are displayed but never averaged in,
because sighted LLM valuations anchor to the market price).

- **CLI**: `python analyze.py TICKER` (add `--no-ai` for the free rule-based card)
- **Web**: FastAPI app with a single admin login, background analysis runs,
  caching, and a Postgres/SQLite run log. Deploys to Railway.

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# 1. generate a cookie secret
python -c "import secrets; print(secrets.token_hex(32))"   # -> SECRET_KEY=...
# 2. hash your admin password
python scripts/hash_password.py                             # -> ADMIN_PASSWORD_HASH=...
# 3. add your AI keys (ANTHROPIC_API_KEY / OPENAI_API_KEY / ...)
#    matching config/analysts.yaml; missing keys just skip that analyst.

# CLI card
python analyze.py AAPL

# Web app (SQLite, http://localhost:8000)
uvicorn webapp.main:app --reload
```

Analyst roster, weights, and per-analyst blind/sighted participation live in
`config/analysts.yaml`; scoring weights and valuation assumptions in
`config/weights.yaml`. The engine (`src/`) is a plain importable package —
the web app is just a wrapper around `src.pipeline.run_full_analysis()`.

## Deploy to Railway (first-timer walkthrough)

1. **Create a GitHub repo and push** this project:
   ```bash
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   `.env` is gitignored — your keys never leave your machine this way.
2. **Create the Railway project**: [railway.app](https://railway.app) → *New
   Project* → *Deploy from GitHub repo* → pick the repo you just pushed.
   Railway reads `railway.json` (build via Nixpacks, start command, `/health`
   healthcheck) automatically.
3. **Add Postgres**: in the project canvas, *Create* → *Database* →
   *PostgreSQL*.
4. **Connect the database**: open your web service → *Variables* → *New
   Variable* → *Add Reference* → select the Postgres service's
   `DATABASE_URL`. (Railway injects it; don't paste it by hand.)
5. **Paste the remaining variables** (same tab):
   - `SECRET_KEY` — output of `python -c "import secrets; print(secrets.token_hex(32))"`
   - `ADMIN_USERNAME` — your login name
   - `ADMIN_PASSWORD_HASH` — output of `python scripts/hash_password.py`
     (paste the whole `$2b$...` string)
   - `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` — your analyst keys
   - optional: `ANALYSIS_CACHE_HOURS` (default 24), `MAX_CONCURRENT_ANALYSES` (default 2)
6. **Deploy**: Railway builds on every push to `main`. Wait for the deploy to
   go green (the `/health` check must pass).
7. **Open the app**: service → *Settings* → *Networking* → *Generate Domain*,
   visit it, and **log in** with your admin credentials.

Tables are created automatically on first startup. Each analysis costs a few
cents in AI calls; the cache and the concurrency cap keep spend bounded.

## Security notes

- Every route except `/login` and `/health` requires the signed session
  cookie (HttpOnly, SameSite=strict — which is also the CSRF defense).
- Login attempts are rate-limited per IP.
- API keys live only in environment variables; they are never rendered,
  stored in the DB, or written to logs.
