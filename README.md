# Wealthsimple Pulse -- The Life-Event Sentinel

A predictive Signal Engine that classifies retail banking users into wealth-tier personas and detects behavioral signals for high-margin product conversion, with a human-in-the-loop traceability UI.

## Personas

| Tier | AUA Range | Signal | Product |
|------|-----------|--------|---------|
| Aspiring Affluent | $50k-$100k | Leapfrog Signal | RRSP Loan (Retirement Accelerator) |
| Sticky Family Leader | $100k-$500k | Liquidity Watchdog | Summit Portfolio + WS Credit Card |
| Generation Nerd | $500k+ | Analyst-in-Pocket | AI Research + Direct Indexing + Private Credit |

### Why the Wealthsimple EQ set

The UI uses the **Wealthsimple EQ** persona names to align with product strategy:

- **Momentum Builder** (Aspiring Affluent): Validates the user's trajectory toward $100k. The Life-Event logic pushes them toward that milestone using the Retirement Accelerator (RRSP Loan).
- **Full-Stack Client** (Sticky Family Leader): Wealthsimple wants these users on everything—Credit Card, Private Equity, and Direct Deposits. This persona is the sticky core of the business.
- **Legacy Architect** (Generation Nerd): Focused on long-term wealth. These users want institutional-grade tools (Direct Indexing, AI Dashboards) to build a multi-generational legacy.

Internal keys (`aspiring_affluent`, `sticky_family_leader`, `generation_nerd`) stay unchanged in code, config, and model files.

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.11+ |
| **Data & ML** | pandas, numpy, scikit-learn, XGBoost, joblib |
| **Synthetic Data** | Faker (profiles), pyarrow (Parquet) |
| **Config** | PyYAML (`config/settings.yaml`) |
| **API** | FastAPI, uvicorn, Pydantic |
| **UI** | Streamlit, Plotly |
| **Storage** | Parquet (raw/processed/experiments), SQLite (feedback) |
| **Testing** | pytest, httpx |

The pipeline is structured so the ML model can be swapped (e.g. to a PyTorch TCN) via the `BaseSignalModel` interface in `src/models/predict.py`.

## Business Rules

### Persona eligibility (AUA thresholds)

- **Not eligible**: AUA &lt; $50k — no product recommendations.
- **Aspiring Affluent**: $50k ≤ AUA &lt; $100k — eligible for Retirement Accelerator (RRSP Loan).
- **Sticky Family Leader**: $100k ≤ AUA &lt; $500k — eligible for Summit Portfolio + WS Credit Card.
- **Generation Nerd**: AUA ≥ $500k — eligible for AI Research + Direct Indexing + Private Credit.

Thresholds are configurable in `config/settings.yaml` under `persona_thresholds`.

### Governance tier (traffic light)

Each recommendation is assigned a governance tier that drives the approval workflow:

| Tier | Condition | Workflow |
|------|-----------|----------|
| **Green** | Confidence &gt; 0.9 and product risk not high | Silent approval or low-friction notification |
| **Amber** | Confidence 0.7–0.9 or medium-risk product | Batch cohort approval (curator can approve the whole cohort) |
| **Red** | Confidence &lt; 0.6, or Summit allocation &gt; 20% of AUA | 1-to-1 human review required |

Product risk: RRSP Loan and AI Research = medium; Summit Portfolio = high. Red is also triggered when illiquidity ratio (e.g. Summit share of AUA) exceeds 20%.

### Macro adjustments

- **Retirement Accelerator (RRSP Loan)**  
  If Bank of Canada prime rate &gt; 5%, confidence is reduced by 10% unless the estimated tax refund offset is ≥ 1.2× the interest cost.

- **Summit Portfolio**  
  If VIX &gt; 25, confidence is reduced (up to 15%) because illiquid PE is riskier in volatile markets.

- **AI Research / Direct Index**  
  If VIX &gt; 25, confidence receives a small boost (3%) — volatile markets increase demand for research and tax-loss harvesting.

Macro inputs (BoC rate, VIX, TSX volatility) are mocked in `src/api/macro_agent.py` and can be replaced with live APIs.

### Active learning (feedback penalty)

Curator approve/reject decisions are stored in SQLite (`data/feedback.db`, table `human_feedback`). For each recommendation type (persona + signal + product), if the rejection rate exceeds 60% and there are at least 3 decisions, future confidence for that type is reduced (up to 15%) so the system learns curator boundaries (e.g. “don’t suggest loans in a recession”).

### Model and surfacing

- Per-persona binary XGBoost classifiers detect signals (e.g. `leapfrog_ready`, `liquidity_warning`, `harvest_opportunity`).
- Confidence threshold and precision target are in `config/settings.yaml` (`model.confidence_threshold`, `model.precision_target`); thresholds are tuned for ~80% precision on holdout data.

### Distance to Upgrade (Status Transition)

- **Momentum Builder:** `gap_to_next_milestone` and `pct_to_milestone` toward Premium ($100k). Cohorts named e.g. "Premium Path: 80% to Milestone."
- **Full-Stack Client:** Same toward Legacy ($500k).
- **Legacy Architect:** "At cap" (no next tier).

Computed in [src/features/wealth.py](src/features/wealth.py); exposed on each hypothesis as `distance_to_upgrade` and in the UI queue/detail.

### Guardrail Cohorts

- **Outlier Sentinel:** Life Inflection Alert when a Legacy Architect shows a large MoM spike in grocery/retail spend; RRSP Loan is not suggested.
- **Cross-Pollination:** Bank-Replacement Lead when WS Credit Card spend ≥90% but Direct Deposit is elsewhere; nudge to move DD.
- **Liquidity Stress:** Summit users with &lt;3 months runway are flagged; curator can pause PE contributions.

See [src/classifier/guardrails.py](src/classifier/guardrails.py).

### Batch Approve and Intent Cohorts

- **POST /batch/approve:** Request body `{ "items": [ { "user_id", "persona_tier", "signal", "product_code", "confidence", "governance_tier" } ], "action": "approved" }`. Records feedback for each item.
- **Intent cohorts:** Premium Leapfrog, Summit Onboarding, Bank-Replacement Lead. Built by [src/classifier/cohort_engine.py](src/classifier/cohort_engine.py). The UI shows Batch Review cards with "Why" and a Global Approve button.

### Dashboard (multi-page)

The Streamlit app is multi-page: the main **Dashboard** (home) and a **Cohort Builder** page (sidebar link or "Open Cohort Builder" in the sidebar). The sidebar includes expandable sections: **Decision Summary & Impact** (AUM unlocked, approvals, progress, decision counts, active learning stats), **Filters** (persona tier, min confidence), **Scenario Planning** (BoC rate, VIX), and **Cohort Builder** (navigate to the dedicated page). **Scenario Planning** is available in three places: the main sidebar, the Cohort Builder page, and each individual client traceability panel; the chosen scenario (BoC, VIX) is stored in session state and used for hypothesis confidence.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Phase 1: Generate synthetic data (1,000 users, 18 months, ~1.8M transactions)
python -m src.data_generator.generator

# Phase 2: Build feature matrix
python -m src.features.pipeline

# Phase 3: Train per-persona signal detectors
python -m src.models.train

# Phase 4: Start the FastAPI service (use --port 8001 if 8000 is blocked on Windows)
python -m uvicorn api.main:app --reload --port 8001

# Phase 5: Launch the curator dashboard
python -m streamlit run ui/app.py
```

## API Endpoints

- `GET /` — Service info
- `GET /health` — Model load status
- `POST /predict` — Submit transactions; returns persona-routed signal hypothesis with governance tier, macro context, and traceability
- `POST /feedback` — Record curator decision (approved/rejected/pending) for active learning
- `POST /batch/approve` — Batch approve or reject a list of cohort members (body: `items` + `action`)
- `GET /feedback/stats` — Aggregate feedback counts and approval rate

## Testing

```bash
pytest tests/ -v
```

## Architecture

```
Transactions --> Feature Engine --> Persona Classifier --> Signal Detector --> Product Hypothesis
                                                                                     |
                                                                          Curator Dashboard
                                                                          (Approve / Reject)
```

## Project Structure

```
src/data_generator/   -- Synthetic data engine (profiles, baseline txns, persona injectors)
src/features/         -- Feature engineering (temporal, categorical, wealth-tier)
src/classifier/       -- Rule-based persona tier classifier
src/models/           -- Per-persona XGBoost signal detectors
api/                  -- FastAPI orchestrator with persona-routed responses
ui/                   -- Streamlit curator dashboard with traceability panels
ui/pages/             -- Cohort Builder page (multi-page app)
config/               -- Settings and thresholds (settings.yaml)
tests/                -- Test suite
```
