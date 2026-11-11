# pgvector-backed RAG Knowledge Base QA Service

English | [简体中文](README.md)

A minimal yet complete RAG service: clean → chunk → embed → rewrite → L2 retrieval → rerank → generate answers with citations. Built from official LangChain components; the vector store is a single Docker container (PostgreSQL + pgvector), no extra retrieval service to deploy.

## Features

- **Text cleaning**: strips HTML/script tags, Markdown images, zero-width and control characters, page-number lines, punctuation-only lines and duplicate lines, while keeping paragraph boundaries
- **Three chunking modes**: by character count (100 / 500 / 1000), by paragraph, by section — character mode with 500 chars is the default
- **Euclidean (L2) retrieval**: nearest neighbours on pgvector; the value returned is a distance, not a similarity, so the field is named `vectorDistance`
- **Real reranking**: DashScope `text-rerank` reorders candidates, with two similarity gates (0.7 / 0.6)
- **Query rewriting**: short or anaphoric questions are expanded before embedding; only the embedding input changes, reranking still scores the original question
- **Cited generation**: forces `[n]` markers, strips out-of-range numbers and regenerates once, counts uncited factual sentences
- **Honest degradation**: rerank failure falls back to vector order, rewrite failure falls back to the original question, and `reranked` / `badReferences` report the truth

## Pipeline

```text
Ingest:
  raw text → clean → chunk (3 modes) → embed → write to langchain_pg_embedding

Ask:
  question → [rewrite if vague] → embed → pgvector L2 search (top 20)
           → rerank → similarity gate (0.7 / 0.6) → build context → LLM answer (citation-checked)
```

## Stack

| Component | Choice |
| --- | --- |
| Runtime | Python 3.12 |
| Web | FastAPI + uvicorn |
| RAG | LangChain (langchain-core / langchain-openai / langchain-text-splitters / langchain-postgres) |
| Vector store | PostgreSQL 16 + pgvector 0.8 (Docker) |
| Models | Any OpenAI-compatible endpoint; defaults to Alibaba Cloud DashScope (`qwen-plus` + `qwen3.7-text-embedding-flash` + `qwen3.7-text-rerank`) |

## Quick start

Requirements: Docker Desktop and Python 3.12.

```bash
# 1. Start the database (creates rag_db and enables the vector extension)
docker compose up -d

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# edit .env and fill in at least OPENAI_API_KEY and RERANK_API_KEY

# 4. Ingest a file (.txt / .md)
python -m app.cli ingest samples/demo.md

# 5. Ask a question
python -m app.cli ask "How large is a chunk by default?"

# 6. Serve over HTTP (port 8000 by default)
uvicorn app.main:app --reload
```

Once running, open <http://127.0.0.1:8000/docs> for interactive API docs.

## Configuration

Everything comes from `.env` or environment variables. `.env` is git-ignored — never commit real keys.

### Database

| Variable | Default | Description |
| --- | --- | --- |
| `PG_HOST` | `127.0.0.1` | Database host |
| `PG_PORT` | `5432` | Port, must match `docker-compose.yml` |
| `PG_DB` | `rag_db` | Database name |
| `PG_USER` / `PG_PASSWORD` | `rag` / `rag_pwd` | Credentials |
| `PG_COLLECTION` | `kb_chunks` | pgvector collection name |

### Models

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_BASE_URL` | DashScope compatible endpoint | Any OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | empty | Required |
| `CHAT_MODEL` | `qwen-plus` | Generation model |
| `EMBEDDING_MODEL` | `qwen3.7-text-embedding-flash` | Embedding model |
| `EMBEDDING_DIMS` | `1024` | Must match the model output, otherwise table creation / search fails on dimension mismatch |
| `RERANK_ENABLED` | `true` | Disable to fall back to pure vector ordering |
| `RERANK_BASE_URL` | `https://dashscope.aliyuncs.com` | Rerank endpoint root |
| `RERANK_API_KEY` | empty | Falls back to `OPENAI_API_KEY` when empty |
| `RERANK_MODEL` | `qwen3.7-text-rerank` | Rerank model |

### Retrieval and generation

| Variable | Default | Description |
| --- | --- | --- |
| `CHUNK_MODE` | `char` | `char` / `paragraph` / `section` |
| `CHUNK_SIZE` | `500` | Chunk size for character mode; only 100 / 500 / 1000 |
| `CHUNK_OVERLAP` | `80` | Overlap between adjacent chunks |
| `RETRIEVE_TOP_N` | `20` | Number of candidates from vector search |
| `RERANK_TOP_K` | `5` | Number of chunks passed to the prompt |
| `SIMILARITY_MODE` | `precise` | `precise` = 0.7 / `broad` = 0.6 |
| `REWRITE_ENABLED` | `true` | Rewrite vague questions before embedding |
| `CHUNK_MAX_CHARS` | `800` | Max chunk length in the prompt; longer chunks are truncated and marked |
| `MAX_CONTEXT_CHARS` | `6000` | Total budget for the reference context |

## Chunking and similarity

Three chunking modes (`chunkMode` / `chunkSize` can override per request on `POST /ingest`):

| Mode | Boundary | Behaviour |
| --- | --- | --- |
| `char` | Character count | Recursive splitter preferring Chinese punctuation; size 100 / 500 / 1000 |
| `paragraph` | Blank line | One paragraph per chunk; oversized paragraphs fall back to character mode |
| `section` | Markdown `#`–`######` | One section per chunk; oversized sections fall back to paragraphs, then characters |

The similarity mode sets the rerank-score gate. Candidates below it are dropped; if none survive, **the LLM is not called** and the service answers "no relevant content found":

| Mode | Gate | When to use |
| --- | --- | --- |
| `precise` (default) | 0.7 | Answers must stick tightly to the material |
| `broad` | 0.6 | Allow looser, still relevant material |

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Configuration overview (models, collection, default chunking and similarity) |
| GET | `/db/status` | DB connectivity, pgvector version, chunk count |
| POST | `/ingest` | Ingest raw text, optionally with a chunking mode |
| POST | `/ingest/file` | Upload a `.txt` / `.md` file |
| POST | `/ask` | Ask a question, optionally with a similarity mode |

### Ingest

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"Chunks are 500 characters by default.","source":"demo.md","chunkMode":"section"}'
```

```json
{"source": "demo.md", "chunks": 1, "chunkMode": "section", "chunkSize": 500, "skipped": false}
```

### Ask

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How large is a chunk by default?","similarityMode":"precise"}'
```

```json
{
  "answer": "Chunks are 500 characters by default [1].",
  "citations": [
    {"index": 1, "source": "demo.md", "chunkIndex": 1, "vectorDistance": 0.9067, "rerankScore": 1.0}
  ],
  "retrieved": 2,
  "reranked": true,
  "similarityMode": "precise",
  "minScore": 0.7,
  "droppedByScore": 1,
  "rewrite": null,
  "uncitedClaims": 0,
  "badReferences": [],
  "citationRetried": false
}
```

Field reference:

- `citationRetried`: whether the model cited non-existent numbers and was regenerated after correction
- `badReferences`: out-of-range citation numbers that remain after regeneration (already stripped)
- `uncitedClaims`: factual sentences without any citation; a high number means retrieval missed the answer
- `droppedByScore`: candidates dropped by the similarity gate
- `rewrite`: rewrite result (`used: false` means the question was not vague and was left untouched)

## CLI

```bash
python -m app.cli db-status
python -m app.cli ingest samples/demo.md --mode section --size 500
python -m app.cli ask "How large is a chunk by default?" --similarity broad
```

## Layout

```text
app/
├─ config.py         environment variables and similarity presets
├─ cleaner.py        text cleaning
├─ chunker.py        three chunking modes
├─ store.py          embeddings and pgvector client
├─ rerank.py         DashScope rerank (degrades on failure)
├─ query_rewrite.py  query rewriting (falls back on failure)
├─ pipeline.py       ingest / ask flow, citation validation
├─ main.py           FastAPI endpoints
└─ cli.py            command line entry
docker/
└─ init.sql          enables the vector extension
docker-compose.yml   pgvector container
```

## Design notes

- **A distance is not a similarity**: `similarity_search_with_score` returns a distance under L2 (smaller is closer), hence the `vectorDistance` name so clients don't mistake it for a 0–1 score
- **DashScope compatible-mode gotcha**: `OpenAIEmbeddings` must be created with `check_embedding_ctx_length=False`, otherwise it tokenizes with tiktoken and sends token arrays, which the endpoint rejects with 400
- **Rerank scores the original question**: the rewritten query is only used for embedding, so rewrite drift cannot skew the ranking
- **The number is the citation**: references are numbered 1..n in rerank order, and `citations[].index` matches the `[n]` markers in the answer
- **Lazy client construction**: the service starts even with incomplete configuration; `/health` just reports degradation instead of refusing to boot over a missing key

## FAQ

**Dimension mismatch on startup**: you changed the embedding model but the table already exists. `EMBEDDING_DIMS` must match the model output; after switching models, clear the collection and re-ingest (vectors from different models are not comparable).

**Rerank returns 403 / 404**: check that `RERANK_MODEL` is available for your account — DashScope model permissions differ. You can also set `RERANK_ENABLED=false` temporarily.

**Retrieval always empty**: check `chunks` in `/db/status` first; then make sure `OPENAI_API_KEY` is not empty (an empty key makes embedding fail silently).

**Port 5432 already in use**: change the port mapping in `docker-compose.yml` and `PG_PORT` in `.env` to the same value.

**Paragraph mode behaves like character mode**: check whether the cleaned text still has blank lines — blank lines are the only boundary for `paragraph` mode.

## License

[MIT](LICENSE)
