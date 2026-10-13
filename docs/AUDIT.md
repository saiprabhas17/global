# GMEE Phased Audit & Acceptance

The dated infrastructure audit below is historical. See [Verification](VERIFICATION.md)
for the subsequently executed source repairs, model/database/browser/pipeline checks
and remaining external/scientific requirements.

**Audit date:** 2026-10-03. **Conclusion:** infrastructure safeguards and validation
mechanisms are implemented; a complete production rollout and scientific
validation are **not claimed**. Main/backend/frontend agents reconcile API and
OpenAPI contracts after their changes merge. This document separates source-level
implementation, actually executed validation and experiments not done.

## Scope and preservation

Infrastructure/CI/docs work is confined to `infra/**`, `.github/workflows/**`,
Dockerfiles/ignores/nginx, `render.yaml`, the allowed Markdown files, the new
backup helper and the two root PowerShell sweeps. No app/models/core/routes,
backend tests, existing evaluation/ML scripts, frontend source/config or
`docs/openapi.json` are edited by this workstream.

Existing user README/CHANGELOG changes and untracked `docs/EVALUATION.md` are
updated granularly, not replaced. The historical publication report artifact is
preserved; its generator belongs to the research workstream. No live database,
backup, migration, container restart or commit is performed by this audit.

## Phases and findings

| Phase | Finding / risk | Response | Evidence level |
|-------|----------------|----------|----------------|
| 0 — inventory | User-modified docs and active local stack must not be clobbered | Read status, preserve artifacts, exclusive file scope; no secret-file inspection | Inspection |
| 1 — boot/deployment | Uvicorn previously started before migrations, with two model workers and a backend source mount | Migration-first fail-closed Docker CMD, one worker, immutable source; app owns lazy models | Source + static policy checks |
| 2 — network/security | Datastores publicly bound; nginx appended spoofable XFF; SPA policy incomplete | Private datastore network/no host ports; loopback dev override; peer-only trust and replacement headers; CSP/body/timeouts/frame denial | Static policy, not penetration test |
| 3 — quality/integration | Frontend used non-locked installs; integration skips could make CI green; nginx runtime was documented as a test runner | `npm ci`, Python/Node test targets, strict JUnit gate, relevant path filters and bounded checks | Config + offline tests |
| 4 — smoke/data safety | Live sweeps posted eval labels/feedback and read DB/secret state | Read-only HTTP sweeps; separate tmpfs-only HTTP smoke with hard deadline and no scientific writes | Offline safety tests; real smoke not run here |
| 5 — operations | No bounded non-overwriting backup helper or explicit isolated restore runbook | `pg_dump` subprocess with safe argv/logs, timeout/exclusive archive/failure cleanup; isolated restore and upgrade procedure | Mocked backup tests; restore not run |
| 6 — product/research | Automatic labels and historical metrics were described as independent gold/publication-grade | Explicit origins, source-level typed/evidence contracts, frozen-data protocol, exploratory historical report and UNPERFORMED ledger | Documentation; research validation not done |

The main upgrade additionally owns `gmee01` typed-edge metadata/parent integrity,
`gmee02` authenticated label origin/split identity, and `gmee03` durable pipeline
ownership/jobs, plus auth/evidence/mutation/operations contracts. Source presence
is not proof the running developer deployment has applied them.

## Validation actually executed in this workstream

| Check | Result / boundary |
|-------|-------------------|
| Static Compose normalization/policy for base, dev override, test and smoke | Passed with temporary non-secret `--env-file`, `--no-env-resolution`, `--no-interpolate`; no containers or datastore connections |
| Dockerfile/nginx/Render/CI policy and local Markdown file links | Static checks passed; not native nginx syntax or hosted Render schema validation |
| `python -m unittest discover -s infra/tests -v` | 20 offline safety tests passed; HTTP and `pg_dump` are mocked, not live E2E/backup execution |
| Windows PowerShell AST parsing | Passed for both root sweeps and shared/parser scripts; no HTTP invocation |
| Targeted Python lint/type/format checks | Final command/result recorded in the workstream handoff; no broader application pass claimed |

Native `nginx -t` and real production-image HTTP smoke are configured as failing
CI gates but **were not executed locally**, to avoid starting/altering the user's
containers. Image build, migrations, actual backup/restore, browser E2E, load and
controlled research experiments are not validated by the above static checks.

## Acceptance checklist — engineering vs product vs research

`Implemented` below means code/config/protocol exists, **not** that every release
gate is checked. Keep unexecuted items open until independent evidence is attached.

### Engineering

- [x] Migration-first startup and single-worker default are enforced in runtime CMD.
- [x] Production backend source bind mount removed; datastore/API host ports absent.
- [x] Dev datastore/API bindings are loopback-only; runtime secret file is required.
- [x] nginx replaces forwarding headers; Uvicorn trusts only the addressed proxy peer.
- [x] SPA CSP permits current Google Fonts and the deployed API origin; framing
  denied, body/API timeout limits and server version hiding declared.
- [x] Dedicated test images/services; frontend locked installation; no npm in nginx.
- [x] CI rejects missing/empty/skipped integration reports; infra/Render path filters.
- [x] Hard-bounded, fixture-only HTTP smoke and read-only live contract sweeps exist.
- [x] Backup refuses overwrite, bounds `pg_dump`, redacts credentials and removes partials.
- [ ] Fresh/upgraded image builds and native nginx/Render configuration validation.
- [ ] Required pgvector/migration integration passes against disposable stores,
  including pre-existing conflicts and provenance-preserving upgrade behavior.
- [ ] Main app's lazy model startup and default-ignore-XFF tests pass; optional
  proxy configuration and external TLS/IP/scheme chain reviewed on target host.
- [ ] Durable ownership/recovery/job/metrics authorization and concurrency tests pass.
- [ ] Isolated restore drill with recorded integrity, RPO/RTO and Redis/Neo4j recovery.
- [ ] Dependency/image/model revision review, secret rotation, least-privilege
  database roles, TLS, access/logging policy and measured capacity before exposure.

### Product

- [x] Intended contracts document authenticated local-corpus checking, citations,
  stance, abstention and typed comparison with no causal-propagation claim.
- [x] Empty corpus is a valid smoke response, not a fabricated demo dataset.
- [x] HTTP smoke exercises registration/login/me and denied admin triggers; no labels.
- [ ] Real fixture smoke passes through production nginx and backend images.
- [ ] Authenticated UI integration and unavailable/empty/abstention/error states
  verified after the source changes merge; no client-calculated scientific result.
- [ ] Browser rendering, keyboard/accessibility, responsive and token/session flows
  exercised in an approved browser E2E setup (no Playwright dependency added here).
- [ ] Runtime docs/OpenAPI/client contracts reconciled by main; metrics mount/body
  and job-recovery semantics confirmed before rollout.

### Research

- [x] Candidate contribution, hypotheses, baselines, ablations and limitations are stated.
- [x] Human/automatic/legacy/test origins and server-controlled identity are explicit.
- [x] Event-disjoint frozen splits, consistent export and reproducibility/licensing
  manifest are specified; automatic labels are not presented as independent gold.
- [x] Historical publication report is classified exploratory and preserved.
- [ ] Independent human annotation, rubric/adjudication, event identity and rights review.
- [ ] Frozen train/dev/test snapshot with overlap tests, model/data hashes and consent.
- [ ] Held-out claim matching/calibration, typed mutation, evidence stance and
  selective-risk experiments, baselines, ablations, uncertainty and error analyses.
- [ ] Generalization, data/model licensing and current literature/novelty comparison.

**All controlled research comparisons above remain UNPERFORMED.** No metric,
novelty claim, fixed human-label significance threshold or production-readiness
promise should be inferred from this audit.

## Release-blocking follow-up record

Attach actual commands, environment/image/schema hashes, redacted logs and dates
for each open engineering/product gate. Attach immutable research manifests,
human provenance, raw predictions and registered analysis for scientific gates.
Do not mark a browser test complete because an HTTP call succeeded, or a restore
complete because a mocked subprocess test passed. Do not use live annotations
or automatic pseudo-label agreement to satisfy scientific acceptance.

Owners reconcile final contracts after agent work returns; see
[Operations](OPERATIONS.md), [API](API.md), [Architecture](../ARCHITECTURE.md),
[Research](RESEARCH.md) and [Evaluation](EVALUATION.md).
