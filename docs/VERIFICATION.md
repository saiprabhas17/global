# GMEE implementation and verification — 13 October 2026

This records changes to `pavanputhra17/GMEE` from base commit
`b4893083dfcef2c2a9e246d019236ca8ebaf2a93`. The changes are in the working tree;
they have not been pushed or deployed to an external service. Validation used
Python 3.11, Node 20, PostgreSQL 16/pgvector, Neo4j 5 and Redis 7 in the cloud
development environment. A passing gate establishes its tested behavior, not
a universal error-free guarantee or a paper acceptance prediction.

## Implemented changes

- The tested Python 3.11/Linux CPU dependencies are recorded in `backend/requirements.lock`; CI and Docker use those constraints and CPU Torch. Other platforms/model swaps need their own validated environment.
- Live graph/corpus routing replaces unconditional static JSON and randomly
  generated analysis. Requests use real authentication, cancellation and errors.
- `gmee04` owns article vector/HNSW and claim full-text/GIN schema. Fresh installs
  and upgraded databases use the same migrated fields. NLP writes document
  vectors in the fenced article transaction; semantic search loads safely.
- Local extraction processes the full article in bounded batches, shares model
  state safely, loads off the event loop and retains source sentences. The
  factual/opinion prompt and structural-score filtering were checked with real
  FLAN-T5 inference. Local summaries and passage stance need no paid provider.
- Quantity/unit normalization preserves original spans and distinguishes
  equivalent quantities from changed amounts. Reported correction/update cues
  and supplied references are disclosed conservatively; unresolved context does
  not become a claim of misinformation or observed propagation.
- Hybrid keyword/vector retrieval returns actual cited corpus passages with
  offsets, timestamps, hashes, source grouping and NLI abstention. Wikipedia
  investigation retrieves revision-bound passages and uses local stance.
- Human review supports a versioned rubric, exact Unicode changed spans,
  contextual relations and independent blinded disagreement review. Label,
  type/span/context agreement remain separate; dissent and revisions persist.
- Neo4j projects articles, sources, claims and real relationship aggregates
  transactionally. Admin reconciliation and small-corpus projection are usable
  without requiring enough claims for topic clustering.
- Admin pipeline controls expose durable run inspection, stage triggers,
  graph sync and expired-work recovery preview. Job leases renew during long
  work; losing a lease stops processing. Public registration stays ordinary;
  an explicit local operator CLI assigns roles to existing accounts.
- Refresh rotation locks the token row against concurrent reuse. Inactive
  accounts cannot log in. Request IDs are installed and SQL parameter logging
  is disabled so development does not log credential hashes/private text.
- Approximate duplicate candidates are checked with exact Jaccard before source
  grouping. Short inputs and retry insertion no longer collapse unrelated text.
- Account dialogs are portaled outside blurred/sticky ancestors. Mobile corpus
  search controls fit their panel. Types, unit fixtures and API contracts were
  repaired without disabling existing strict quality gates.
- Legacy evaluators refuse the invalid LIAR extraction target, require frozen
  FEVER evidence and refuse empty mutation data. Documentation and historical
  publication claims are reconciled with the deployed models and human protocol.

## Executed checks

| Check | Result |
|-------|--------|
| Backend unit tests and coverage gate | 517 passing; 83.07% application coverage (70% gate) |
| Ruff and strict mypy on app/tests/scripts | Pass; no ignored gate or lowered threshold |
| Frontend component/API tests | 116 passing |
| Frontend coverage gates | Pass; 94.03% statement/line coverage, 81.64% branches |
| Frontend ESLint and TypeScript/Vite production build | Pass |
| Real pretrained model smoke | Three passing, zero skips: MPNet vectors/determinism, DeBERTa support/contradiction, FLAN source-sentence extraction |
| Real PostgreSQL integration | Nine passing, zero skips, including schema/API and refresh concurrency |
| Completely fresh database | Entire migration chain to `gmee04`, then all nine integration cases pass; temporary database removed |
| Infrastructure checks | 20 offline tests and static Compose/Docker/CI/Markdown policy checks pass |
| Live readiness/authentication | All three stores ready; registration/login/me, ordinary-user admin denial and typed comparison pass |
| Live NLP/evolution pipeline | Four explicitly synthetic articles → 32 real-model-extracted claims → eight clusters → 41 relationships; Neo4j sync succeeds |
| Live corpus/investigation API | Four article vectors searchable; three actual evidence passages returned; graph has four nodes/six article links; fourteen read endpoints plus search/check/sync pass |
| Real Chromium UI smoke | All ten tabs, login, corpus evidence, unit-equivalent comparison, admin recovery preview/sync; zero page errors or server 500s; desktop and mobile viewports |

The synthetic pipeline demonstrates execution and integration only. Its
clusters/links/stances are not independently annotated accuracy results. No
scientific labels were written. Its temporary corpus, graph nodes and operator
accounts are removed after validation. Original stores and existing user data
are preserved. Logs and screenshots are retained outside the checkout under
`/workspace/gmee-validation` for this environment.

The pretrained snapshots used here resolve to:

| Capability | Model | Snapshot revision |
|------------|-------|-------------------|
| Retrieval | sentence-transformers/all-mpnet-base-v2 | e8c3b32edf5434bc2275fc9bab85f82640a19130 |
| Extraction | google/flan-t5-base | 7bcac572ce56db69c1ea7c8af255c5d7c9672fc2 |
| Stance | cross-encoder/nli-deberta-v3-base | 6c749ce3425cd33b46d187e45b92bbf96ee12ec7 |

`python scripts/prepare_models.py` records resolved snapshots; use approved
model revisions/local snapshot paths for frozen research runs. Run real model
checks with `pytest -m model -o addopts=''`; unit tests exclude model downloads.

## Local PostgreSQL password recovery

The Windows launcher now accepts `-RepairPostgresPassword` for a retained
PostgreSQL volume whose stored role password differs from the current local
configuration. Ordinary startup never rotates credentials. Explicit recovery
uses the container-local socket, preserves the existing `.env` and database
contents, and resets the backend/frontend restart state before checking readiness.
Startup failures now display the backend's recent logs directly.

The reported authentication failure was reproduced against a disposable
PostgreSQL 16 container with SCRAM host authentication. The repair ran through
the PowerShell launcher and real `psql`: the current password then authenticated,
the old password was rejected, and an existing sentinel row was unchanged.
Five controlled launcher scenarios pass: normal startup, explicit recovery,
backend failure, unavailable PostgreSQL, and unsuccessful repair. PowerShell
syntax, the 20 infrastructure tests, and static deployment checks also pass.
The launcher scenarios are required in the Windows CI job.

This validates the credential repair and launcher control flow. The user's
Windows Docker instance is not remotely accessible from this workspace. A full
container rebuild here remains blocked by build-container resolution of the
cloud proxy; it is not claimed as a successful Windows startup test.

## External and scientific requirements

Live Wikipedia and the default BBC RSS HTTPS destinations were denied by the
cloud proxy. Wikipedia's 17 controlled-upstream
tests pass, including source revisions, historical cutoff, bounds and model
abstention; real network retrieval remains unverified. The environment draft
preserves existing domains and adds `en.wikipedia.org`. Review/save that network
change in environment settings and publish the environment before retrying the
live checks. `feeds.bbci.co.uk` is also included for the default public RSS feed.
A saved draft does not apply the policy.

Optional Reddit, NewsAPI, Hugging Face hosted inference and Anthropic integrations
need their provider credentials/allowed network destinations. Their mocked
failure/contract coverage is not a successful live provider call. Default local
NLP works without those keys. Enable the scheduler explicitly after configuring
the sources; it defaults off.

Complete production container deployment, TLS exposure, capacity/load, full
disaster restore and comprehensive accessibility/security audits are separate
unperformed deployment studies. Browser smoke and static Compose validation
do not substitute for them.

For publication, independently annotate real licensed event data, freeze
event-disjoint splits, compare current baselines/ablations and report grouped
uncertainty and error analysis. The annotation/evaluation tooling is implemented;
human gold, empirical novelty and paper-worthy results remain to be established.
See [Evaluation](EVALUATION.md), [Research](RESEARCH.md) and
[Publication plan](../PUBLICATION_PLAN.md).
