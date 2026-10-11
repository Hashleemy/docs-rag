# Model card and governance notes

A short, honest record of what this system does, which models it uses, what data goes where,
and what it must not be trusted with. Written in the spirit of the NIST AI Risk Management
Framework (Govern / Map / Measure / Manage).

## Purpose

Answer questions about a user's own documents (PDF, Markdown, text, images) with citations
to the exact passages used. Personal project; not connected to any employer's data or systems.

## Models

| Role | Model | Where it runs | Why |
|---|---|---|---|
| Embeddings (turn text into vectors) | `BAAI/bge-small-en-v1.5` via fastembed | Locally on the CPU | Free, private, small (384 dimensions), good enough for English documents |
| Answers and image descriptions | Claude (`LLM_MODEL`, default `claude-opus-5-5`) | Anthropic API | Strong reading comprehension and vision; swap to `claude-haiku-5-5` for low-cost experimentation |

## Data flow (what leaves the machine)

1. Uploaded files are stored under `storage/uploads/`. Only on this machine.
2. Chunks and their vectors are stored in Chroma under `storage/chroma/`. Only on this machine.
3. For every question, the question plus the top-k retrieved excerpts are sent to the
   Anthropic API. Nothing else from the index is sent.
4. For every image ingested, the image bytes are sent to the Anthropic API once, to be described.
5. Chroma's anonymous usage telemetry is switched off (`ANONYMIZED_TELEMETRY=False`).

Do not ingest documents you are not allowed to send to a third-party API.

## Security controls

- Secrets only via environment variables or `.env` (git-ignored); never in code or images.
- Upload allow-list by file extension; size cap (`MAX_UPLOAD_MB`); file names are reduced to a
  safe base name, so path traversal (`../../etc/passwd`) is impossible.
- The system prompt treats retrieved text as **untrusted data** and tells the model not to follow
  instructions found inside documents (prompt-injection mitigation, not elimination).
- Container runs as a non-root user; dependencies are pinned by `uv.lock` and `package-lock.json`.
- **No authentication on the API.** Run it on localhost or behind an authenticating proxy; add
  auth before exposing it to the internet.

## Observability

- Every request is logged as one JSON line (method, path, status, milliseconds).
- Every answer records model, latency, input tokens, output tokens, and the number of excerpts.
- `/api/health` reports the model in use and the number of indexed chunks.
- Each response carries an `x-response-time-ms` header.

## Evaluation

`python -m app.cli eval` runs a golden set of questions (`eval/golden.json`) and reports the
retrieval hit rate: whether the expected document (and page, when given) appears in the top k.
It fails below 80 %. This measures retrieval only; answer quality is reviewed by hand using the
citations.

## Known limitations

- Scanned PDFs without a text layer are skipped (no OCR).
- Answers can still be wrong or incomplete; the citations exist so a person can check.
- English-only embedding model.
- Single process, single node; the index is a local folder.
