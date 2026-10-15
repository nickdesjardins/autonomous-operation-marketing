# Container image for the API (docker compose --profile full, or any container host).
# The public demo on Render uses the python runtime instead (render.yaml); both run the same
# command. The service runs as an unprivileged user and keeps one worker: the demo budget and
# the model chooser live in that process's memory.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --system --uid 10001 --no-create-home app

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY sql ./sql

USER app

EXPOSE 8000

# /livez answers without touching the database, so the health check never wakes a serverless one.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, sys, urllib.request; port = os.environ.get('PORT', '8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/livez').status == 200 else 1)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
