# ---------- Stage 1: build the React UI ----------
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---------- Stage 2: Python runtime ----------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    EMBEDDING_CACHE_DIR=/app/.cache/fastembed \
    PYTHONUNBUFFERED=1

# Dependencies first so this layer is cached between code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY --from=web /web/dist ./web/dist

# Download the embedding model at build time: the container starts instantly and works offline
RUN uv run --no-sync python -c "from app.factory import warm_embeddings; warm_embeddings()"

# Run as a non-root user; storage/ is where Chroma and uploads live (mount a volume there)
RUN useradd --create-home appuser && mkdir -p /app/storage && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1
CMD ["/app/.venv/bin/uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
