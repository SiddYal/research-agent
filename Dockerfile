FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LOG_JSON=true

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY alembic.ini .
COPY alembic ./alembic
COPY research_agent ./research_agent

RUN useradd --create-home app && mkdir -p /app/data/reports && chown -R app /app/data
USER app

EXPOSE 8000
# Migrations run on startup (init_db) as well, so this works with any DATABASE_URL.
# --proxy-headers so per-IP rate limiting sees the client, not the platform's proxy.
CMD ["sh", "-c", "uvicorn research_agent.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
