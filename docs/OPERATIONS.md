# GMEE Operations & Deployment

**Scope:** engineering runbook, not a production-readiness certificate. Commands
that start containers, migrate, back up or restore are operator procedures and
were **not executed against the developer's running services** during this audit.
See [Audit](AUDIT.md) for exact validation and remaining gates.

## Deployment modes and rationale

| Configuration | Purpose | State / host exposure |
|---------------|---------|-----------------------|
| `infra/docker-compose.yml` | Immutable production-like images | Persistent named volumes; only SPA on `127.0.0.1:3000` |
| Add `infra/docker-compose.dev.yml` | Host-local tools | Datastores/API bound to loopback only; still no backend source mount |
| `infra/docker-compose.test.yml` | Unit/component tests | Dedicated Python/Node test targets; no runtime `.env` or databases |
| `infra/docker-compose.smoke.yml` alone | Real HTTP integration fixture | tmpfs datastores, no host ports/fixed container names/runtime `.env` |
| `render.yaml` | Managed API and separate static SPA | Operator-provided managed stores; capacity and proxy chain must be reviewed |

Keep the existing Compose project, database names and volume identities when
upgrading. Do not use `down --volumes` on a real deployment. Changing
`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` or Neo4j's initialization auth
variables **does not rotate credentials in existing volumes**. The template's
fresh-install values are not an instruction to rename an existing database.

## Required runtime configuration

Create private root `.env` from `infra/runtime.env.example` **only if absent**.
Fill empty passwords and a unique JWT secret of at least 32 characters. Use
URL-safe generated PostgreSQL credentials (random hex is suitable), because
Compose assembles the in-network URL. For other passwords, percent-encoding and
URL construction need deliberate review rather than pasting raw reserved chars.
Never commit, echo, upload or include the secret file in a support report.

Always pass `--env-file .env` for runtime commands. The backend's service
`env_file` is also explicitly required. Compose overrides datastore addresses
with private service names; host-local development must separately configure
URLs with loopback and the development ports. Optional provider keys are only
needed for chosen collectors/LLMs; account for upstream quotas, costs, and the
privacy implications of sending article text to a provider.

```bash
# Operator action; not part of static validation.
docker compose --env-file .env -f infra/docker-compose.yml up -d --build
# Local development, never a public-host deployment:
docker compose --env-file .env -f infra/docker-compose.yml -f infra/docker-compose.dev.yml up -d --build
```

The base deployment is intentionally not a public TLS termination solution.
Place a managed TLS gateway before loopback nginx; configure certificates,
request logging, DNS and access controls before public exposure. Do not simply
publish PostgreSQL, Neo4j, Redis or the API to all interfaces.

## Migration-first startup and upgrade sequence

1. Identify the existing schema revision, volume/project identity and credentials
   through authorized operator tooling; do not copy secrets into tickets.
2. Take a protected backup and prove a restore on a separate isolated target.
3. Review the migrations against that restored snapshot before the maintenance
   window; preserve historical data and resolve conflicts explicitly.
4. Stop new ingestion/labeling writes using the application's supported controls.
5. Deploy the reviewed images. Docker CMD executes `alembic upgrade head` and
   only then `exec uvicorn ... --workers 1`; migration errors prevent API startup.
6. Check `/api/v1/health/ready`, then authenticated API contracts and durable job
   state. Enable scheduled work explicitly only after these checks succeed.

| Revision | Upgrade intent / operator caution |
|----------|-----------------------------------|
| `gmee01` | Nullable typed mutation evidence and one `EVOLVED_FROM` parent per child. Historical multiple-parent conflicts fail preflight; they are not silently deleted. NULL legacy evidence is not backfilled scientific proof. |
| `gmee02` | Origins (`human`, `automatic`, `legacy`, `test`), authenticated annotator IDs, label history, canonical pairs, splits and event identity. Existing votes remain conservative `legacy`; migration does not turn old handles into human gold. |
| `gmee03` | Pipeline ownership/lease and durable job records. |
| `gmee04` | Migrated article vectors/HNSW and claim full-text/GIN search; preserves prior manually populated vectors. |

Alembic migrations are not a replica-coordination mechanism. This deployment is
single-process/single-API-container by default. A future multi-replica rollout
must use an operator-controlled, one-shot migration phase before replicas start,
plus proven fencing/recovery semantics. Never solve a failed upgrade by stamping
head, deleting historical relationships, or blindly downgrading provenance.

## Proxy trust and SPA policy

- nginx **replaces** `X-Forwarded-For`, `X-Real-IP`, scheme/host/port headers from
  its own connection and clears `Forwarded` and alternate forwarding chains.
- The production proxy network is `172.30.47.0/24`; nginx is `.10`, backend `.20`.
  Uvicorn `FORWARDED_ALLOW_IPS=172.30.47.10` trusts that peer only, not `*` or the
  entire Docker installation. If this subnet conflicts, update the network,
  addresses and allowlist together and rerun validation.
- The application defaults to ignoring raw XFF; any optional app-level trust
  configuration must be restricted to verified proxy networks. Uvicorn's socket
  trust and a raw-header parser are separate layers; do not trust the same chain
  twice or accept an internet client's first XFF value as identity.
- Render retains loopback-only trust until provider-confirmed proxy addresses
  and the whole forwarding chain are configured. Accurate client-IP attribution
  is a remaining deployment gate, not a reason to enable wildcard trust.
- TLS gateways need an explicit trusted nginx real-IP/scheme policy if original
  client IP/HTTPS attribution is required. The supplied nginx replaces headers
  using its **immediate** socket peer and scheme, not arbitrary gateway headers.
- CSP allows scripts from self, current Google Fonts CSS/font origins, and inline
  React styles. `connect-src 'self'` includes same-origin `/api/v1`; `/api` is not
  a separate CSP host. Render's separate static site additionally allows its
  declared backend HTTPS origin. Update both Vite build URL and CSP/CORS when
  changing that origin.
- Framing is denied by CSP and `X-Frame-Options: DENY`; MIME sniffing is denied.
  nginx hides its version, caps request bodies at 1 MiB, connects within 5 s,
  sends within 30 s and waits for API response data up to 60 s. These are bounds,
  not a measured latency SLO. Long work belongs in durable jobs, not one HTTP call.

## Models, capacity and durable operations

One worker avoids a separate embedding/NLI model copy per Uvicorn process. Main
application code owns lazy initialization; deployment does not eagerly warm
models. A named cache is writable by the non-root backend user, but cached model
weights/revisions must still be recorded for reproducible research. First-use
model downloads need network/disk space and may exceed request latency budgets.
Image builds fetch packages and spaCy weights; runtime local-corpus checking does
not imply a network-free first model load.

The scheduler defaults off, including Render. The free Render plan is retained
as a **demo configuration**, not a validated ML RAM budget. Do not change to a
paid plan without an explicit cost/capacity decision. Measure cold/warm RSS,
query p50/p95/p99, pool pressure, bounded graph sizes and job backlog before
promising a public SLO. Current capacity/load experiments are **UNPERFORMED**.

Main upgrade operator APIs are admin-only job inspection/recovery and metrics:
`/api/v1/operations/jobs`, `/api/v1/operations/recover`, and the final mounted
metrics route. Review the generated OpenAPI for exact request bodies. Recover
only demonstrably expired ownership; do not clear Redis locks or requeue active
work merely because a request timed out. Durable rows are the diagnostic source;
Redis election alone is not exactly-once execution. Crash, lease-expiry,
long-running renewal, concurrent triggers and stale-owner fencing need isolated
integration tests before HA claims. Never run recovery sweeps against a live
instance as a deployment smoke test.

## Safe database backup

`backend/scripts/backup_database.py` launches `pg_dump` with no password/URL in
argv, no shell, no prompt, a bounded timeout (default 300 s, maximum 3600 s),
exclusive file creation and failure cleanup. It validates the custom archive
header and never loads `.env` or logs subprocess stderr. It does **not** restore,
migrate, encrypt, schedule or test the archive. PostgreSQL archives can contain
user data, sessions, annotations and source text: protect them as sensitive data.

Prerequisites: a `pg_dump` client compatible with the server (PostgreSQL 16 for
this stack), sufficient SELECT rights, and a secure operator network path. The
backend runtime does not bundle an arbitrary distro's potentially older client.
Use a private operator runner/tunnel; do not open a public database port to make
backups easier. Provision credentials via secure process/libpq environment or an
operator-managed `PGPASSFILE`; do not put passwords in CLI arguments.

```bash
# Manual examples only. Create/protect the output directory first.
# POSTGRES_URL must already exist in this process environment:
python backend/scripts/backup_database.py --output backups/gmee-20261003.dump --timeout-seconds 300
# Alternatively, explicitly configure PGHOST/PGPORT/PGUSER/PGDATABASE + credentials:
python backend/scripts/backup_database.py --use-libpq-environment --output backups/gmee-20261003-libpq.dump --timeout-seconds 300
```

SQLAlchemy `postgresql+asyncpg` URLs are normalized to libpq variables; supported
URL query options are `sslmode`, `sslrootcert`, `sslcert`, `sslkey`. For a URL with
asyncpg-only parameters, use explicit libpq mode instead. Existing files are
never overwritten. POSIX files are created mode `0600`; Windows requires
appropriate NTFS ACLs. Encrypt at rest/offsite with approved tooling, record a
checksum and retention policy, and exclude archives from source/images.

### Restore drill — isolated target only

1. Provision a **separate PostgreSQL 16 + pgvector instance**, a new restore DB,
   and isolated credentials/network. Reconfigure libpq environment to that
   instance; never reuse the production source URL as the restore target.
2. Create the empty DB. If it already exists, stop and select a new name rather
   than dropping/cleaning it.
3. Restore with error-stop and no owner/ACL import, using the isolated credentials.

```bash
# Not executed during this audit; target host is configured externally.
createdb --maintenance-db=postgres -- gmee_restore_drill_20261003
pg_restore --exit-on-error --no-owner --no-acl --dbname gmee_restore_drill_20261003 backups/gmee-20261003.dump
```

4. Verify archive checksum, schema revision, pgvector extension/indexes, key row
   counts, annotation origins/history and job records against the backup manifest.
   Disable scheduled collection/provider calls in the drill, then test readiness
   and representative read/auth contracts. Apply a candidate migration only to
   this restored instance, not the source.
5. Record restore duration, integrity checks, retention/access policy and agreed
   RPO/RTO. Delete only the isolated drill after review. **Restore execution and
   disaster-recovery performance are UNPERFORMED** for this upgrade.

PostgreSQL is authoritative; Neo4j must have a separate tested backup or a proven
rebuild from relational state. Decide how Redis loss affects revocations,
rate-limits and ownership before calling a PostgreSQL-only dump a full recovery.

## Validation and CI strategy

```bash
# Static only; requires PyYAML + modern Docker Compose, never reads runtime .env:
python infra/scripts/validate_deployment.py
python -m unittest discover -s infra/tests -v
powershell.exe -NoProfile -File infra/scripts/validate_powershell.ps1
```

The validator uses a temporary non-secret `--env-file`, `--no-interpolate` and
`--no-env-resolution`, and never prints resolved configuration. Do not replace it
with plain `docker compose config` or `config --environment` in logs: those can
expand runtime credentials. Native `nginx -t` and hosted Render schema validation
are separate gates; textual policy checks do not prove runtime configuration.

CI checks backend lint/types/unit coverage (70%), real migrated pgvector
integration with a zero-skip/zero-empty JUnit gate, frontend locked installs,
lint/types/build/component coverage, static deployment policies, PowerShell parse
and **HTTP** smoke through production images. Connectivity failure must fail the
integration job, not be treated as an optional local skip. Path filters include
`infra/**` and `render.yaml`; configure branch protection carefully, because a
required path-filtered workflow can remain pending on unrelated-only changes.
A repository-wide required-check dispatcher is a repository-settings decision.

### Disposable HTTP smoke (manual fixture procedure, not run here)

Use the smoke file **alone** and an unused project name. It has no host ports,
root secret env files or persistent database volumes; credentials are deliberately
fixture-only. The smoke process requires two fixture acknowledgements, checks
zero articles/claims/pairs and kills its child at 240 s (plus bounded termination).
Readiness retries and per-request timeouts are also bounded. Missing APIs, auth
failures, unavailable stores, malformed shapes or incorrect typed changes fail;
there are no success-by-skip paths.

```bash
docker compose --project-name gmee-smoke-20261003 --env-file infra/fixture.env -f infra/docker-compose.smoke.yml build
docker compose --project-name gmee-smoke-20261003 --env-file infra/fixture.env -f infra/docker-compose.smoke.yml run --rm --no-deps frontend nginx -t
docker compose --project-name gmee-smoke-20261003 --env-file infra/fixture.env -f infra/docker-compose.smoke.yml up --abort-on-container-exit --exit-code-from smoke --timeout 15
# Cleanup ONLY the project just created above, never the runtime deployment:
docker compose --project-name gmee-smoke-20261003 --env-file infra/fixture.env -f infra/docker-compose.smoke.yml down --volumes --remove-orphans --timeout 15
```

The fixture registers/logs in an ordinary user, checks `/auth/me`, denies all
admin triggers for anonymous/user requests, probes readiness/empty corpus,
checks evidence and eval auth gates, and calls the real mutation API twice
(identical control + numeric change). No `/eval/label`, verdict feedback, corpus
seed or authorized trigger is called. Offline safety tests mock HTTP/`pg_dump`;
they are not evidence that the real container smoke passed.

`verify_endpoints.ps1` and `_ui_sweep.ps1` are read-only HTTP checks, use an
optional `GMEE_VERIFY_TOKEN` process environment variable, honor Base/timeouts,
redact bodies/tokens and return nonzero on failure. Reports are opt-in with
`-ReportPath` and refuse overwrite. They do not render a browser, register users
or create scientific labels. Browser interaction/accessibility E2E requires a
separate approved dependency/test setup; no Playwright package was added.

Current local runtime/model verification is recorded in [Verification](VERIFICATION.md).
Register an ordinary account, then use `python scripts/set_user_role.py --email
ACCOUNT --role admin` from backend with private operator database configuration
to provision an administrator. The UI exposes bounded pipeline controls and
recovery preview to administrators; public registration cannot grant the role.
