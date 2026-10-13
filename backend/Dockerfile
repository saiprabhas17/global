FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.lock alembic.ini ./
COPY app ./app
COPY alembic ./alembic
COPY scripts ./scripts
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-deps 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install -c requirements.lock -e . && \
    python -m pip install -c requirements.lock 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'

# Tests run in a Python image with dev dependencies, never in the runtime SPA.
FROM base AS test
RUN python -m pip install -c requirements.lock '.[dev]'
COPY tests ./tests
CMD ["pytest", "-m", "not integration and not model", "--strict-markers"]

FROM base AS production
RUN useradd --create-home --uid 10001 gmee && \
    mkdir -p /home/gmee/.cache && \
    chown -R gmee:gmee /app /home/gmee
USER gmee
ENV HOME=/home/gmee
EXPOSE 8000

# Fail closed on migration errors; one worker avoids duplicate ML weights.
# Uvicorn trusts only an explicitly configured proxy peer (loopback by default).
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1 --proxy-headers --forwarded-allow-ips \"${FORWARDED_ALLOW_IPS:-127.0.0.1}\""]
