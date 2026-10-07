# Single image for API, dashboard, bootstrap (seed + train) and MLflow server.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# libgomp1: OpenMP runtime required by LightGBM / XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY api ./api
COPY training ./training
COPY dashboard ./dashboard
COPY data/synthetic ./data/synthetic
COPY models ./models
COPY reports ./reports
COPY docker/*.sh ./docker/
COPY pyproject.toml README.md ./
RUN chmod +x docker/*.sh && mkdir -p data/processed

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000 8501 5000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
