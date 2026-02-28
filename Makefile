# Wealthsimple Pulse - Life Event Predictor
# Run from project root

.PHONY: simulate_experiment run-api run-ui test

simulate_experiment:
	python -m scripts.simulate_experiment

run-api:
	python -m uvicorn api.main:app --reload --port 8001

run-ui:
	python -m streamlit run ui/app.py

test:
	pytest tests/ -v
