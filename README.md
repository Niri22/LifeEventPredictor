# Wealthsimple Pulse -- The Life-Event Sentinel

A predictive Signal Engine that classifies retail banking users into wealth-tier personas and detects behavioral signals for high-margin product conversion, with a human-in-the-loop traceability UI.

## Personas

| Tier | AUA Range | Signal | Product |
|------|-----------|--------|---------|
| Aspiring Affluent | $50k-$100k | Leapfrog Signal | RRSP Loan (Retirement Accelerator) |
| Sticky Family Leader | $100k-$500k | Liquidity Watchdog | Summit Portfolio + WS Credit Card |
| Generation Nerd | $500k+ | Analyst-in-Pocket | AI Research + Direct Indexing + Private Credit |

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

# Phase 4: Start the FastAPI service
uvicorn api.main:app --reload

# Phase 5: Launch the curator dashboard
streamlit run ui/app.py
```

## API Endpoints

- `GET /` -- Service info
- `GET /health` -- Model load status
- `POST /predict` -- Submit transactions, receive persona-routed signal hypothesis with full traceability

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
config/               -- Settings and thresholds (settings.yaml)
tests/                -- Test suite
```
