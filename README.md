# 📈 Stock Evaluator

> **An intelligent stock analysis platform combining deterministic financial models with a multi-LLM analyst ensemble.**

Stock Evaluator is designed to provide investors with transparent, research-driven stock evaluations by combining traditional financial analysis with independent AI analysts.

Unlike AI tools that simply summarize news or make unsupported recommendations, Stock Evaluator separates **objective financial analysis** from **subjective AI reasoning**, allowing users to see both the numbers and multiple investment perspectives.

---

# 🚀 Features

## 📊 Rule-Based Financial Analysis

Every company is first evaluated using a deterministic financial engine.

The engine analyzes:

- Financial ratios
- Profitability
- Growth
- Balance sheet strength
- Cash flow
- Capital efficiency
- Valuation metrics
- Business quality
- Risk factors

Because this stage is entirely rule-based, identical financial data will always produce identical results.

---

## 🤖 Multi-LLM Analyst Ensemble

Instead of relying on a single AI model, Stock Evaluator supports multiple analyst personas across different providers.

Supported providers include:

- OpenAI
- Anthropic Claude
- Additional providers can easily be added through the modular adapter system

Each analyst independently reviews the company before contributing their insights.

---

## 👨‍💼 Dual-Pass Analysis

One of the project's defining features is its **dual-pass evaluation process**.

### Blind Analysis

AI analysts first evaluate the company **without knowing its market price**.

This prevents valuation anchoring and produces an independent estimate of intrinsic value.

The blind valuations are combined into the consensus fair value.

---

### Sighted Review

After the blind valuation, analysts receive the complete investment packet, including the current stock price.

They then act as critics by discussing:

- Risks
- Weaknesses
- Competitive concerns
- Valuation disagreements
- Business quality
- Industry outlook

These sighted opinions are displayed separately and are **never averaged into the fair value estimate**, preserving objectivity.

---

# 📈 Analysis Pipeline

```
Financial Data
        │
        ▼
Data Collection
        │
        ▼
Ratio Engine
        │
        ▼
Business Archetype Classification
        │
        ▼
Pillar Scoring
        │
        ▼
Red Flag Detection
        │
        ▼
Rule-Based Fair Value
        │
        ▼
Blind AI Analyst Consensus
        │
        ▼
Sighted AI Critiques
        │
        ▼
Final Investment Report
```

---

# 🧠 Business Archetypes

Rather than evaluating every company with identical assumptions, the engine first classifies the business into an archetype.

Examples include:

- Growth companies
- Mature businesses
- Value stocks
- Asset-heavy firms
- Cash-generating businesses

Each archetype uses different scoring weights and valuation assumptions to better reflect how businesses should be analyzed.

---

# ⚠️ Red Flag Detection

Before producing a recommendation, the engine evaluates potential warning signs such as:

- Weak balance sheet
- Excessive leverage
- Declining profitability
- Poor cash flow
- Margin deterioration
- Financial instability

These red flags can override otherwise attractive scores.

---

# 📊 Fair Value Estimation

Intrinsic value combines:

- Rule-based valuation models
- Financial quality assessment
- Archetype-specific assumptions
- Blind AI analyst consensus

This creates a valuation that is grounded in financial fundamentals while benefiting from diverse AI reasoning.

---

# 🌐 Web Dashboard

Stock Evaluator includes a modern FastAPI web application featuring:

- Secure administrator login
- Stock search
- Cached analysis results
- Background analysis jobs
- Historical run logging
- Interactive reports
- Financial glossary
- PostgreSQL or SQLite support

---

# 💻 Command Line Interface

Analyze a stock directly from your terminal.

```bash
python analyze.py AAPL
```

Run without AI:

```bash
python analyze.py AAPL --no-ai
```

Perfect for fast deterministic financial reports.

---

# 🏗 Architecture

```
                 User
                  │
        ┌─────────▼─────────┐
        │   FastAPI Web UI  │
        └─────────┬─────────┘
                  │
        Analysis Pipeline
                  │
        ┌─────────▼─────────┐
        │ Data Collection   │
        ├───────────────────┤
        │ Ratio Engine      │
        ├───────────────────┤
        │ Archetype Model   │
        ├───────────────────┤
        │ Scoring Engine    │
        ├───────────────────┤
        │ Fair Value Model  │
        ├───────────────────┤
        │ AI Ensemble       │
        └─────────┬─────────┘
                  │
          Final Investment Report
```

---

# 🛠 Technology Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| Backend | FastAPI |
| Database | PostgreSQL / SQLite |
| ORM | SQLAlchemy |
| Market Data | yfinance |
| AI Providers | OpenAI, Anthropic |
| Templates | Jinja2 |
| Authentication | bcrypt + signed sessions |
| Deployment | Railway |

---

# 📁 Project Structure

```
Stock_Evaluator/

├── analyze.py
├── requirements.txt
├── config/
│   ├── analysts.yaml
│   ├── glossary.yaml
│   └── weights.yaml
│
├── scripts/
│   └── hash_password.py
│
├── src/
│   ├── data_fetcher.py
│   ├── ratio_engine.py
│   ├── scoring.py
│   ├── archetype_classifier.py
│   ├── fair_value.py
│   ├── ensemble.py
│   ├── report.py
│   └── pipeline.py
│
└── webapp/
    ├── auth.py
    ├── main.py
    ├── service.py
    └── templates/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Stock_Evaluator.git

cd Stock_Evaluator
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Configure environment variables

```bash
cp .env.example .env
```

Add your API keys and administrator credentials.

---

# 🚀 Running the Application

CLI

```bash
python analyze.py AAPL
```

Web application

```bash
uvicorn webapp.main:app --reload
```

Visit:

```
http://localhost:8000
```

---

# ☁️ Deployment

The project is configured for deployment on **Railway**.

Features include:

- Automatic deployments
- PostgreSQL integration
- Environment variable management
- Health checks
- Secure session handling

Simply connect your GitHub repository, configure the required environment variables, and deploy.

---

# 🔒 Security

Stock Evaluator follows several security best practices:

- Secure administrator authentication
- Password hashing with bcrypt
- Signed session cookies
- HTTP-only authentication
- Rate-limited login attempts
- Environment-based API key storage
- No API keys stored in the database

---

# 🎯 Design Philosophy

Stock Evaluator is built around one central principle:

> **AI should assist investment research—not replace disciplined financial analysis.**

Financial statements, ratios, and valuation models provide the objective foundation, while AI contributes qualitative insights, alternative perspectives, and risk assessment.

By separating deterministic analysis from subjective reasoning, the platform delivers transparent and explainable investment evaluations.

---

# 🛣 Roadmap

- Additional AI providers
- Portfolio analysis
- Watchlists
- Historical backtesting
- Earnings transcript analysis
- News sentiment integration
- Sector comparison dashboards
- DCF customization
- PDF report exports
- API endpoints for external integrations

---

# 🤝 Contributing

Contributions are welcome!

Whether you're improving valuation models, adding AI providers, refining scoring algorithms, or enhancing the user interface, pull requests and discussions are encouraged.

---

# 📄 License

This project is currently unlicensed.

Consider adding an open-source license (MIT, Apache 2.0, or GPL) before public release.

---

## Disclaimer

Stock Evaluator is an educational and research tool.

It does **not** provide financial, legal, or investment advice. Always conduct your own due diligence before making investment decisions.
