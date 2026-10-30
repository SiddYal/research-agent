# Autonomous Research Agent

Ask a research question; get back a report in which **every factual sentence is cited and every
claim behind it has been cross-checked against independent sources**, plus metrics that score
the report automatically.

```
question ─▶ plan ─▶ (you approve) ─▶ search ─▶ clean + sanitize + chunk + embed
        ─▶ hybrid retrieval + rerank ─▶ extract atomic claims ─▶ verify across sources
        ─▶ write from verified claims only ─▶ evaluate ─▶ report + metrics
```

Built with LangGraph, FastAPI, PostgreSQL + pgvector, and OpenAI or Claude.

## Why

LLM "research" tools tend to blend retrieved text with model memory, and cite sources that don't
say what's claimed. This agent separates the steps that cause that:

- **The writer never sees raw web text.** It receives only claims that passed verification, each
  tagged with the sources that support it. Contested and unverified claims are listed separately,
  and the report doesn't rely on them.
- **Verification is cross-source.** Every extracted claim is checked against passages from
  *other* sources and labelled verified, uncertain or contradicted, weighting each source by
  quality. Uncorroborated claims trigger a targeted follow-up search before the verdict is final.
- **Scraped pages are untrusted input.** Prompt-injection text is stripped before it reaches any
  model, and sources that attempt it are down-weighted as witnesses.
- **Every run is measured**: citation coverage, source diversity, hallucination risk (LLM-as-judge
  against the verified claims), claim verdicts, latency and cost.

## Quick start

**Easiest (Windows):** double-click **`start.bat`**. On first run it installs everything and opens
`.env` for you to paste your OpenAI API key; run it again and the app opens in your browser at
<http://localhost:8000>. On macOS or Linux, run `./start.sh`.

In the app you:

1. pick the AI model and type your question;
2. review the research plan: edit, remove or add subquestions, then approve (or reject);
3. watch each step run, then read the report, with scores, clickable citations, and
   copy/download buttons. Past research stays in the *Recent research* list.

**Manual setup**

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then set OPENAI_API_KEY (or ANTHROPIC_API_KEY)
```

**CLI.** You review the plan before research starts (approve, edit the subquestions, or reject):

```bash
python -m research_agent "How do pgvector, Pinecone, and Weaviate compare for RAG workloads?"
```

`--yes` skips the review, `--out report.md` saves the report, `--list` shows past jobs, and
`--resume <job_id>` retries a failed job from its last checkpoint.

**API + web UI:**

```bash
uvicorn research_agent.api.main:app --reload
```

Then open <http://localhost:8000> (interactive API docs at `/docs`).

**Docker (API + Postgres/pgvector):**

```bash
docker compose up --build
```

## API

| Method | Path | |
|---|---|---|
| `GET` | `/models` | AI models the server can use (a provider appears once its API key is set) |
| `POST` | `/research` | `{"question": "...", "model": "openai/gpt-4.1-mini", "auto_approve": false}` → `202 {job_id}` |
| `GET` | `/research/{id}` | status, plan, per-node timeline, metrics, cost |
| `POST` | `/research/{id}/approve` | `{"approved": true, "plan": <optional edited plan>}` |
| `POST` | `/research/{id}/resume` | retry a failed job from its last checkpoint |
| `GET` | `/research/{id}/report` | Markdown; `?format=json` adds sources, claims and verdicts |
| `GET` | `/research` | recent jobs |

Job states: `queued → running → awaiting_approval → running → completed`, or `failed` /
`rejected`. Every response carries an `x-request-id` header that also appears on every log line.

## Architecture

Details are in [docs/architecture.md](docs/architecture.md). In short:

- **LangGraph** orchestrates ten nodes over a single typed `ResearchState`, checkpointed after
  every node. That's what makes jobs pausable (human approval), resumable after a failure without
  re-running finished nodes, and recoverable after a process restart.
- **Retrieval** cleans pages with trafilatura, chunks them by section (≈450 tokens, 50-token
  overlap), stores embeddings in **pgvector**, and fuses **BM25 + vector** results with reciprocal
  rank fusion before an LLM rerank.
- **Storage**: PostgreSQL + pgvector in Docker and production; SQLite with no setup locally.
  Alembic owns the schema.
- **Providers**: OpenAI by default (`gpt-4.1-mini`, `text-embedding-3-small`); set
  `LLM_PROVIDER=anthropic` to use Claude (`claude-haiku-5-5` by default). Search is DuckDuckGo
  with no key needed, or Tavily if `TAVILY_API_KEY` is set.

```
research_agent/
  graph.py        workflow + checkpointer        service.py   job lifecycle
  state.py        ResearchState                  llm.py       providers, cost metering, budgets
  nodes/          planner, search, retrieve, verify, writer, evaluate
  tools/          search, fetch/clean, sanitize, chunking, retrieval, source quality
  db/             SQLAlchemy models, sessions, vector store
  api/            FastAPI app + static web UI
  evaluation.py   report metrics                 cli.py       command line
```

## Metrics

Each completed job stores, next to its report:

| Metric | Meaning |
|---|---|
| citation coverage | share of report sentences carrying at least one valid citation |
| hallucination risk | share of sentences an LLM judge finds unsupported by the verified claims |
| source diversity | unique domains ÷ sources cited |
| claims verified / uncertain / contradicted | verifier verdicts |
| injection flags | paragraphs removed by the sanitizer |
| latency, LLM calls, tokens, cost | from the per-node timeline |

Run the reference questions with `python scripts/generate_examples.py`; reports and metrics land in
[examples/](examples/). Results from the checked-in runs (`gpt-4.1-mini`, DuckDuckGo, 15 sources each):

| Question | Citation coverage | Hallucination risk | Claims verified / uncertain / contradicted | Cost | Time |
|---|---|---|---|---|---|
| [pgvector vs Pinecone vs Weaviate for RAG](examples/how-do-pgvector-pinecone-and-weaviate-compare-for-retrieval-.md) | 79% | 0% | 19 / 3 / 2 | $0.046 | 166 s |
| [LangGraph vs CrewAI vs AutoGen](examples/how-do-langgraph-crewai-and-autogen-differ-as-frameworks-for.md) | 73% | 0% | 21 / 1 / 2 | $0.040 | 164 s |
| [AI data center power demand](examples/what-is-driving-the-growth-in-ai-data-center-power-demand-an.md) | 79% | 12% | 24 / 5 / 0 | $0.049 | 203 s |

The uncited sentences are mostly topic sentences and the conclusion's limitations; the judge counts
those strictly.

## Configuration

All settings are environment variables (or `.env`); see [config.py](research_agent/config.py)
for the full list.

| Variable | Default | |
|---|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | – | at least one is required |
| `LLM_PROVIDER` | auto | `openai` or `anthropic` |
| `DATABASE_URL` | `sqlite:///./data/research.db` | e.g. `postgresql+psycopg://…` |
| `MAX_SOURCES` | 15 | sources fetched per job |
| `MAX_VERIFICATION_ROUNDS` | 1 | follow-up search rounds for uncorroborated claims |
| `RERANKER` | `llm` | `llm` or `none` |
| `RATE_LIMIT` | `5/minute` | per IP, on `POST /research` |
| `MAX_JOB_COST_USD` | 0.50 | a job stops if it spends more |
| `DAILY_COST_CAP_USD` | 2.00 | new jobs are refused beyond this |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs offline with deterministic fakes for the LLM, search and web. It covers every node
in isolation, the full graph (pause and resume, failure recovery, restart survival, budget stop),
the API (including rate limiting and the cost cap), and two planted-source checks: a source
stating a false fact is flagged **contradicted** and never reaches the report body, and a
prompt-injection string is stripped before any model sees it. CI runs the suite against both
SQLite and Postgres + pgvector, and builds the Docker image.

## Deployment

A [render.yaml](render.yaml) blueprint deploys the Docker image to Render. Point `DATABASE_URL`
at a Neon Postgres database; the migration enables pgvector. Railway works the same way.

## Security posture

This is a portfolio demo, not a hardened service. There is **no authentication**. Abuse is
bounded by per-IP rate limiting, a per-job spend limit and a hard daily spend cap. Scraped content
is treated as untrusted (sanitized, delimited, never shown to the writer), and the web UI sanitizes
rendered Markdown. It isn't built for production traffic or sensitive questions.

## Roadmap

The core build (v0.1 → v1.0) is implemented here. Next is the extension plan: explicit
reflect/retry loops, connectors and auth, agent evaluation in CI, ETL, ambiguity handling, full
tracing and EKS. The original pre-prototype is kept in [prototype/](prototype/) for reference.
