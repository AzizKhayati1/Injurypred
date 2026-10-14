.PHONY: help install web web-serve data validate train evaluate experiments monitor test lint api dashboard docker-up docker-down clean

PY ?= python

help:            ## show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:         ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

data:            ## (re)generate the synthetic dataset
	$(PY) -m src.data.synthetic

validate:        ## run data validation
	$(PY) -m src.data.validation

train:           ## train, select, evaluate and register the model (MLflow)
	$(PY) -m training.train

evaluate:        ## full evaluation report for the registered model
	$(PY) -m training.evaluate

experiments:     ## workload-representation and class-imbalance experiments
	$(PY) -m training.experiments.workload_representations
	$(PY) -m training.experiments.imbalance_strategies

monitor:         ## drift / monitoring report for the current season
	$(PY) -m src.monitoring.drift

test:            ## run the test suite
	$(PY) -m pytest

lint:            ## static checks
	ruff check .

api:             ## run the API locally on :8000
	uvicorn api.main:app --reload --port 8000

dashboard:       ## run the dashboard locally on :8501 (uses the API if running, else in-process)
	streamlit run dashboard/app.py

mlflow-ui:       ## browse local MLflow runs on :5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

web:             ## export pre-computed data for the static (Netlify) dashboard → web/data
	$(PY) -m scripts.export_static

web-serve:       ## preview the static site on :8080 with Netlify-like routing
	$(PY) -m scripts.serve_web

docker-up:       ## full stack: API, dashboard, PostgreSQL, MLflow
	docker compose up --build

docker-down:
	docker compose down -v

clean:
	rm -rf data/processed/*.parquet .pytest_cache .ruff_cache
