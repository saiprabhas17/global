# GMEE Backend & Pipeline Engine

FastAPI/SQLAlchemy/Alembic service for the **Global Misinformation Evolution
Engine**. It collects attributable articles, extracts claims, searches a local
corpus, describes typed text changes and exposes operational/evaluation controls.
It is a research prototype: corpus agreement is not truth, inferred graph edges
are not observed copying, and scientific experiments remain **UNPERFORMED**.

See [system architecture](../ARCHITECTURE.md), [API](../docs/API.md),
[Operations](../docs/OPERATIONS.md), [Audit](../docs/AUDIT.md) and
[Research](../docs/RESEARCH.md) for rationale and release gates.

## Stack and source map

- Python 3.11+, FastAPI/Starlette/Uvicorn, async SQLAlchemy 2 and Alembic
- PostgreSQL 16 + pgvector as authoritative storage; Neo4j graph projection;
  Redis for security/coordination state
- all-mpnet-base-v2 (768-d) retrieval, spaCy `en_core_web_sm`, local stance/
  extraction models and optional configured providers; record exact revisions
  before a reproducible evaluation

| Path | Role |
|------|------|
| `app/api/v1/` | Auth, health, corpus, verdicts, graph, eval, alerts and pipeline/operations contracts |
| `app/core/` | Settings, security, scheduler and ownership/limiting behavior |
| `app/db/`, `app/models/`, `app/repositories/` | Async clients, relational source of truth and data access |
| `app/services/` | Collection/preprocessing/NLP, typed changes, local evidence and eval |
| `alembic/` | Schema migration history; authoritative field/index definitions |
| `scripts/` | Operator batch/evaluation tools and safe database backup helper |
| `tests/` | Mocked/SQLite unit tests and required real-pgvector integration |

## Local setup (operator actions)

```bash
cd backend
python -m venv .venv
# POSIX: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --no-deps 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements.lock -e ".[dev]"
python -m spacy download en_core_web_sm
```

Configure private environment locally; never print or commit secrets. Runtime
Compose uses root `.env` derived from `infra/runtime.env.example`; it is required
and supplies in-network datastore URLs. Host-local execution instead requires
`POSTGRES_URL` with loopback/dev PostgreSQL port 55432, Neo4j loopback port 7687,
Redis loopback port 6379, `CORS_ORIGINS`, unique JWT secret (at least 32 chars),
and actual datastore credentials. Initialization-variable changes do not rotate
credentials in existing volumes. Start with `ENABLE_SCHEDULER=false` until
providers, migrations and job safety have been reviewed.

Apply schema before serving the local process:

```bash
python scripts/prepare_models.py
alembic upgrade head
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Local Swagger/ReDoc/OpenAPI: `/docs`, `/redoc`, `/openapi.json`. Liveness:
`/api/v1/health`; readiness: `/api/v1/health/ready` (all stores must be OK).
A successful liveness response is not readiness or loaded-model availability.
`requirements.lock` records the tested Python 3.11/Linux CPU dependency set;
CI and Docker apply it. Validate a new environment before changing pins.

## Container deployment

From repository root:

```bash
# Local host tools only; datastore/API ports bind to loopback.
docker compose --env-file .env -f infra/docker-compose.yml -f infra/docker-compose.dev.yml up -d --build
# Base deployment: datastore/API host ports absent, SPA on loopback for a TLS gateway.
docker compose --env-file .env -f infra/docker-compose.yml up -d --build
```

Production Docker CMD runs `alembic upgrade head && exec uvicorn ... --workers 1`.
A migration failure stops startup; no source bind mount overwrites the image.
One worker avoids duplicate model RAM; the app owns lazy initialization and the
non-root user has a writable model cache. Lazy first use may require model
network downloads and warm-up. In a future replicated deployment, run one
coordinated migration phase before API replicas rather than racing per-replica
DDL. Do not stamp head or remove historical data to force an upgrade through.

`gmee01` adds typed edge evidence/one-parent integrity, `gmee02` adds authenticated
label origin/history and split/event identity, `gmee03` adds durable pipeline ownership/jobs, and `gmee04` adds migrated article
vectors/HNSW and claim full-text/GIN search. Review a restored isolated snapshot before applying
these to existing data. Full instructions: [upgrade runbook](../docs/OPERATIONS.md).

## API and epistemic contracts

- Registration returns the existing token pair and assigns role `user`.
  Login/refresh return bearer credentials; `/auth/me` requires current auth.
  Public registration cannot make an admin.
- Authenticated `POST /verdicts/check` accepts bounded `claim_text`, optional
  ISO `as_of` and evidence `limit`; searches the **stored local corpus**, returns
  article/claim citations, passages, stance, syndication grouping, warnings and
  `INSUFFICIENT_EVIDENCE` when appropriate. Missing evidence is not falsity.
- Authenticated `POST /graph/mutation/compare` accepts `older_text`, `newer_text`
  and optional timestamps; returns typed changes/spans/version and
  `observed_propagation: false`. It does not persist synthetic scientific data.
- `/eval/next`, `/eval/label`, `/eval/progress` require auth. Annotator identity
  is server-controlled; human/automatic/legacy/test origins and revisions are
  separate. Exports/split assignment/reports require admin; do not label fixtures
  or automated guesses as human gold.
- Pipeline triggers and durable jobs/recovery/metrics are admin-only. Redis
  election alone is not an exactly-once guarantee; check final mounted contracts
  and concurrency tests before enabling public/high-availability operations.
- Raw cosine and live verdict fields are uncalibrated heuristics unless a
  versioned, human-held-out calibration experiment explicitly establishes the
  target. Claim relatedness calibration would still not be truth calibration.

The main agent regenerates `docs/openapi.json`; a checked-in snapshot must be
reconciled with source/client docs before release, not hand-edited here.

## Tests and quality gates

```bash
# From backend, with dev dependencies installed:
ruff check app tests scripts
mypy app tests scripts
pytest -m "not integration and not model" --strict-markers --cov=app --cov-fail-under=70
pytest -m model -o addopts='' --strict-markers
```

Unit tests use SQLite shims and mocks; they cannot prove pgvector SQL/index,
PostgreSQL constraints, migrations or multi-worker lease behavior. Integration
must target an explicitly disposable PostgreSQL+pgvector database, never a live
scientific corpus. CI supplies and migrates that service, then:

```bash
pytest tests/integration -m integration -o addopts='' --strict-markers --junitxml=integration-results.xml
python ../infra/scripts/require_integration_results.py integration-results.xml
```

CI sets `CI=true`/required-integration mode and rejects connectivity failures,
missing/empty reports and **any skipped integration case**. The final guard is
independent of optional-local-skip behavior. Backend coverage gate is 70%.

Dedicated container unit tests (root commands):

```bash
docker compose --env-file infra/fixture.env -f infra/docker-compose.test.yml run --rm --build backend-tests
docker compose --env-file infra/fixture.env -f infra/docker-compose.test.yml run --rm --build frontend-tests
```

The Node test target is separate from nginx runtime. Deployment CI also checks
Compose/proxy/Render policies, parses PowerShell and runs bounded real **HTTP**
smoke in tmpfs fixtures, including auth/readiness/empty corpus and typed comparison.
No evaluation labels, feedback or authorized triggers are written by smoke.
This is not browser E2E or an accuracy/performance benchmark.

## Evaluation and historical artifacts

[Evaluation](../docs/EVALUATION.md) preserves the sampling/automatic diagnostic
workflow, with an explicit weak-label interpretation. Existing
`eval_report.json` / `eval_publication_report.md` are exploratory artifacts,
not publication-ready independently annotated results. Do not run sampling or
automatic labeling against a live dataset as a verification sweep.

The scientific protocol requires separately reviewed authenticated human labels,
curated event identity, frozen train/dev/test records, train-fitted methods,
dev-selected calibration/thresholds, untouched test analysis, independent mutation
gold, licensed snapshots and reproducibility manifests. The presence of a harness
or a passing toy test does not mean those experiments occurred. See
[Research](../docs/RESEARCH.md) for hypotheses/baselines/ablations and the
**UNPERFORMED** ledger.

## Backup and recovery

`scripts/backup_database.py` is a bounded, non-overwriting custom `pg_dump`
wrapper. It never loads `.env`, puts credentials in command arguments or logs
stderr. Configure a compatible client and secure process/libpq credentials; it
creates sensitive plaintext archives, not encrypted or restore-verified backups.
Use [Operations](../docs/OPERATIONS.md) for isolated restore procedures, retention,
Windows ACLs and Neo4j/Redis recovery boundaries. No live backup/restore was run
as part of this infrastructure validation.

Current executed validation is recorded in [Verification](../docs/VERIFICATION.md).
