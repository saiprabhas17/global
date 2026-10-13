# GMEE REST API Reference

Base URL: `http://localhost:8000/api/v1`

- **Auth:** JWT bearer. `POST /auth/login` and `POST /auth/refresh` issue
  access + refresh tokens; send `Authorization: Bearer <access_token>`.
- **Roles:** `user`, `researcher` and `admin`. All `*/trigger` endpoints require `admin`.
- **Rate limits:** login 10/min · register 5 per 5 min · triggers ≈4/min ·
  other endpoints 120/min (sliding window; fail-open if the limiter store is
  down). 429 is returned with `Retry-After`.
- Machine-readable spec: [`docs/openapi.json`](openapi.json) — also served
  live at `/docs` (Swagger UI) and `/openapi.json`.

---

## Health — `health.py`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | public | Liveness: process up, no external checks. |
| GET | `/health/ready` | public | Readiness: pings Postgres, Neo4j, Redis; returns per-service status. Powers the dashboard's connection state. |

## Dashboard — `dashboard.py`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/dashboard` | public | Aggregate snapshot for the dashboard UI: services health, corpus counts (articles / claims / embedded claims / entities), NLP claim-status distribution, latest clustering run, Neo4j node & relationship counts, Redis memory stats. Read-only; safe to poll. |

## Auth — `auth.py`

| Method | Path | Auth | Body | Description |
|--------|------|------|------|-------------|
| POST | `/auth/register` | public | `UserCreate` | Create ordinary user `{email, password, full_name?}` → `TokenResponse`. Audited. Rate-limited. |
| POST | `/auth/login` | public | `LoginRequest` | `{email, password}` → `TokenResponse {access_token, refresh_token, token_type}`. Audited. Rate-limited. |
| POST | `/auth/refresh` | public | `RefreshRequest` | Exchange a valid refresh token for a new token pair. |
| POST | `/auth/logout` | bearer | `LogoutRequest` | Revokes the supplied refresh token (added to the Redis JWT blocklist). |
| GET | `/auth/me` | bearer | — | Current user as `UserResponse {id, email, role, full_name?, is_active, created_at}`. |

## Collection — `collection.py`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/collection/status` | bearer | Collector states, article counts, last-run info. |
| POST | `/collection/trigger` | admin | Kick off a collection cycle across enabled sources (RSS / Reddit / NewsAPI). |

## Preprocessing — `preprocessing.py`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/preprocessing/status` | bearer | Pending/processed article counts and pipeline health. |
| POST | `/preprocessing/trigger` | admin | Run cleaning, language filtering, near-duplicate removal, metadata extraction over pending articles. |

## NLP — `nlp.py`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/nlp/status` | bearer | Claim extraction/embedding progress and model availability. |
| POST | `/nlp/trigger` | admin | Extract claims (LLM), embed them (all-mpnet-base-v2 → pgvector), extract entities (spaCy). |

## Evolution — `evolution.py`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/evolution/status` | bearer | Last clustering run, cluster count, relationship totals. |
| GET | `/evolution/clusters` | bearer | Cluster listing with member claims for graph exploration. |
| POST | `/evolution/trigger` | admin | Force an evolution cycle (`force=true` bypasses the debounce guard). Clusters topics, detects `EVOLVED_FROM` / `SIMILAR_TO` relationships, syncs to Neo4j. |

## Corpus — `corpus.py`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/corpus/stats` | public | Corpus totals: articles, embedded count, outlet count, date range, NLP status counts. |
| GET | `/corpus/articles` | public | Paginated articles (`q`, `domain`, `limit`, `offset`) plus domain facet counts. |
| GET | `/corpus/articles/recent` | public | Newest ingested articles — feeds the Live Ingest Stream panel. |
| GET | `/corpus/search` | public | Semantic search: the query is embedded and matched against article vectors (pgvector). |
| GET | `/corpus/graph/story-clusters` | public | Multi-outlet story clusters from the Neo4j SIMILAR graph, ranked by degree. |

## Verdicts — `verdicts.py`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/verdicts` | public | Paginated claim verdicts; `band=` narrows to one verdict band. |
| GET | `/verdicts/stats` | public | Band distribution and per-outlet credibility table. |
| GET | `/verdicts/summary` | public | Engine configuration (weights, band thresholds, pooling rule) and distribution. |
| GET | `/verdicts/leaderboard` | public | Outlets ranked by credibility. |
| GET | `/verdicts/game/mutations` | public | Mutation chains (≥ `min_versions` versions) for the Mutation DNA view. |
| GET | `/verdicts/game/claim` | public | Random claim for the Fact-or-Fake arcade. |
| GET | `/verdicts/{claim_id}` | public | Verdict detail: probability, rationale, evidence, checked neighbours, grounded entities. |
| POST | `/verdicts/feedback` | public | Human vote `{claim_id, vote: AGREE\|DISAGREE, corrected_verdict?, comment?}` → 201 (upsert per client). |

## Graph — `graph.py`, `lineage.py`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/graph/full` | public | Article-level SIMILAR graph (Neo4j) with node/edge counts. |
| GET | `/graph/claims` | public | Claim network. Edges come from persisted `claim_relationships`; a bounded pgvector KNN (`LATERAL`) is used only when the selected claims have no stored links — `edge_source` reports which path served the request. |
| GET | `/graph/timeline` | public | Story clusters ordered **newest-first** for the 3D timeline tunnel (index 0 = latest development; diving deeper reads back in time to the original first report). With `?article_id=`, only clusters within 3 SIMILAR hops of that article are returned (`focused: true`) — the graph→timeline drill-down. |
| GET | `/graph/scoops` | public | Multi-outlet scoop races: first publisher plus exact publish-time lag per outlet. |
| GET | `/graph/simulate` | public | Cascade sandbox: `hub`, `p`, `max_depth` → deterministic reach, expected reach, `r_effective`. |
| GET | `/graph/lineage/{claim_id}` | public | EVOLVED_FROM component around a claim as a time-ordered version chain with word-level typed diffs. `404` when the claim has no lineage, `400` for a bad UUID. |

## Alerts — `alerts.py`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/alerts` | public | Stateless 24h snapshot: ingestion / disputed / mutation spikes versus the previous day, plus the raw counters. Always answers, even on a cold corpus. |
| GET | `/alerts/feed` | public | Persisted alerts from the `alerts` table (deduplicated on kind + subject). Filters: `limit`, `kind`, `severity`, `include_acknowledged`; returns unacknowledged tallies by severity and kind. |
| POST | `/alerts/evaluate` | admin | Run one persistent evaluation pass (`window_hours` 1–48): corroboration, contradiction, mutation and cluster-burst detection with dedup upserts. |
| POST | `/alerts/{alert_id}/acknowledge` | admin | Operator acknowledgement; idempotent — the first acknowledgement timestamp sticks. |

## Evaluation — `eval.py`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/eval/next` | bearer | Next unlabeled gold pair for `?annotator=` — **blinded**: only the two claim texts and their outlets are returned, never the similarity score, bucket or verdict. |
| POST | `/eval/label` | bearer | Record (or update) a label `{pair_id, annotator, label: SAME_STORY\|EVOLVED\|DISTINCT}` → 201. Labeled in-app on the **Eval Lab** tab (`#/dashboard/eval`): blinded pair, keyboard votes (1/2/3), per-bucket coverage grid, Cohen's kappa chips. |
| GET | `/eval/progress` | bearer | Labeling coverage per similarity bucket, per-annotator counts and Cohen's kappa between annotators over shared pairs. |

---

## Typical flow

```
1. POST /auth/register        # ordinary account; operator CLI grants admin
2. POST /auth/login           # get tokens
3. POST /collection/trigger   # ingest articles            (admin)
4. POST /preprocessing/trigger# clean + dedupe             (admin)
5. POST /nlp/trigger          # claims + embeddings        (admin)
6. POST /evolution/trigger    # clusters + mutation graph  (admin)
7. GET  /dashboard            # snapshot for the UI
```

Steps 3–6 also run automatically on the scheduler; triggers exist for
on-demand control.

## Error shape

Validation errors return FastAPI's standard 422 envelope:

```json
{ "detail": [ { "loc": ["body", "email"], "msg": "...", "type": "..." } ] }
```

Auth failures return `401 {"detail": "..."}`, insufficient role `403`,
rate limit `429` (+ `Retry-After`).


## Current investigation/review/operations additions

The generated [OpenAPI](openapi.json) is authoritative for exact bodies and response fields.

| Method | Path | Auth | Behavior |
|--------|------|------|----------|
| POST | `/verdicts/check` | bearer | Bounded hybrid keyword/vector corpus retrieval, actual cited passages, stance, timestamps, offsets/hashes, canonical grouping and abstention. |
| POST | `/verdicts/check-external` | bearer | Fixed Wikipedia revision-bound full passages with local NLI; independent of an Anthropic key. Returns a clear unavailable error for upstream failures. |
| POST | `/graph/mutation/compare` | bearer | Quantity/unit-normalized typed spans, timestamps, conservative contextual relation, optional supplied context/reference and optional summary. References remain unverified. |
| GET | `/eval/rubric` | bearer | Versioned labels, types, exact Unicode span and context instructions. |
| GET | `/eval/review/next` | admin | Blinded independent disagreement review; dissent is retained. |
| GET | `/operations/jobs` | admin | Bounded durable recent-job inspection with optional job-type filter. |
| GET | `/operations/metrics` | admin | Stage/job counts and expired lease count. |
| POST | `/operations/recover` | admin | `{stage: nlp or preprocessing, dry_run: true by default}`; only expired leased work is eligible. |
| POST | `/evolution/graph/sync` | admin | Bounded transactional reconciliation of persisted articles/claims/relationships into Neo4j, independent of clustering size. |

`/eval/label` also accepts `mutation_types`, `context_relation`, `changed_spans`
and notes. Annotator identity/origin are assigned by the server. A span contains
side `a`/`b`, start/end Unicode code-point offsets, exact text and mutation type.
Empty spans explicitly record a completed no-span review; omission means unknown.
This workflow does not invent independently adjudicated gold.
