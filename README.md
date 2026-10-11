# docs-rag

Ask questions about your own documents and get answers with citations.
A small, complete retrieval-augmented generation (RAG) application:

- **Ingestion pipeline** for PDF, Markdown, text, **and images** (described by a vision model),
  chunked with LangChain and stored in a **Chroma** vector database with local embeddings.
- **RAG chain** (LangChain Expression Language) over **Claude**, with numbered citations and a
  prompt that treats document text as untrusted.
- **FastAPI** service with JSON logging, latency and token usage on every answer.
- **React** (Vite) front end served by the same container.
- **Docker** multi-stage image (non-root, health check) and **docker compose**.
- **GitHub Actions** CI: lint, tests (offline, fake model), UI build, Docker build, push to GHCR.
- **Kubernetes** manifests, a **retrieval eval**, and a **model card** (`GOVERNANCE.md`).

```
            ┌──────────────┐    upload     ┌────────────────────────────────────────┐
  browser ─▶│ React (Vite) │──────────────▶│ FastAPI                                 │
            └──────────────┘    ask        │  /api/ingest ─▶ load ─▶ split ─▶ embed ─┼─▶ Chroma
                                           │  /api/ask    ─▶ retrieve top-k ─▶ Claude ┼─▶ answer + citations
                                           └────────────────────────────────────────┘
                                                   images ─▶ Claude vision ─▶ text ─▶ embed
```

## Quick start

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), Node 22+, an Anthropic API key.

```bash
cp .env.example .env            # put your ANTHROPIC_API_KEY in .env
uv sync                         # install Python dependencies into .venv
uv run pytest                   # 14 tests, offline, no API key needed

# index a document, ask a question (command line)
curl -L -o data/NIST.AI.100-1.pdf https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
uv run python -m app.cli ingest data/ README.md
uv run python -m app.cli ask "What are the four functions of the AI RMF Core?"

# run the API + UI
(cd web && npm ci && npm run build)
uv run uvicorn app.main:create_app --factory --reload
open http://127.0.0.1:8000
```

For front-end development run `npm run dev` inside `web/`; the Vite dev server proxies `/api`
to the FastAPI process on port 8000.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | | status, model, embedding model, chunk count |
| GET | `/api/sources` | | indexed files and chunk counts |
| POST | `/api/ingest` | multipart `files[]` (pdf, md, txt, png, jpg, jpeg, webp, gif) | files indexed, skipped, chunk count |
| POST | `/api/ask` | `{"question": "..."}` | answer, numbered sources (file, page, score, snippet), token usage, model, latency |

## How it works

1. **Load.** PDFs become one document per page (pypdf); text files one document; images are
   sent to Claude with a "describe this for a search index" prompt and the description is indexed.
2. **Split.** `RecursiveCharacterTextSplitter`, 800 characters with 120 overlap, so a sentence cut
   at a boundary still appears whole in one chunk.
3. **Embed and store.** `bge-small-en-v1.5` (local, free) vectors in Chroma with cosine distance.
   Chunk ids are a hash of file, page, chunk number and content, and a file's old chunks are
   deleted before it is re-ingested, so re-ingesting never duplicates.
4. **Retrieve and answer.** Top-k chunks are numbered and placed in the prompt; Claude answers with
   `[n]` citations and is told the excerpts are untrusted data. The response includes the
   excerpts, their scores, the model's token usage and the latency.

## Evaluation

```bash
uv run python -m app.cli eval --golden eval/golden.json
```

Prints HIT/MISS per question and the retrieval hit rate at k; exits non-zero below 80 %.
Edit `eval/golden.json` to match your own documents.

## Docker

```bash
docker build -t docs-rag:local .
docker run --rm -p 8000:8000 --env-file .env -v "$PWD/storage:/app/storage" docs-rag:local
# or
docker compose up --build
```

The image is built in two stages (Node builds the UI, Python runs the API), downloads the
embedding model at build time, and runs as a non-root user with a health check.

## Kubernetes (local)

```bash
kubectl create secret generic docs-rag-secrets --from-literal=ANTHROPIC_API_KEY=sk-ant-...
kubectl apply -f k8s/deployment.yaml
kubectl get pods -w
```

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request: `ruff` lint, `pytest` with the
fake model (no secrets in CI), the Vite build, and a Docker build. On pushes to `main` the image
is published to GitHub Container Registry as `ghcr.io/<owner>/docs-rag`.

## Configuration

All settings live in `app/config.py` and can be overridden by environment variables or `.env`:
`LLM_MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MAX_UPLOAD_MB`,
`CHROMA_DIR`, `UPLOAD_DIR`, `FAKE_LLM`.

## Governance

See [GOVERNANCE.md](GOVERNANCE.md): models used, what data leaves the machine, security controls,
observability, evaluation, known limitations.

## Roadmap

- Hybrid search (BM25 + vectors) and a reranker
- Streaming answers to the UI
- OCR for scanned PDFs
- Authentication in front of the API
