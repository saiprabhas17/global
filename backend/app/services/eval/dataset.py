"""Versioned, deterministic human-gold exports and split/leakage validation."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import select, text
from sqlalchemy.orm import aliased

from app.models.article import Article
from app.models.claim import Claim, ClaimEntity
from app.models.eval import EvalPair, EvalPairLabel
from app.services.eval.annotation import ChangedSpan, ContextRelation, validate_spans
from app.services.eval.features import FEATURE_VERSION, entity_jaccard, lexical_jaccard
from app.services.eval.provenance import (
    CONSENSUS_RULE,
    LABELS,
    ORIGINS,
    human_identity,
    origin_counts,
    pair_consensus,
)

SCHEMA_VERSION = "gmee-research-v1"
SPLITS = ("train", "dev", "test")
POSITIVE_LABELS = ("SAME_STORY", "EVOLVED")


class DatasetError(ValueError):
    """An actionable refusal, not a substitute result."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def records_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(canonical_json(r) + "\n" for r in sorted(records, key=lambda r: str(r["pair_id"]))).encode("utf-8")


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.replace(tzinfo=value.tzinfo or UTC).astimezone(UTC).isoformat()


def vote_record(label: EvalPairLabel) -> dict[str, Any]:
    return {
        "id": str(label.id),
        "pair_id": str(label.pair_id),
        "annotator": label.annotator,
        "annotator_user_id": str(label.annotator_user_id) if label.annotator_user_id else None,
        "origin": label.origin,
        "label": label.label,
        "mutation_types": label.mutation_types,
        "notes": label.notes,
        "created_at": iso(label.created_at),
        "updated_at": iso(label.updated_at),
        "provenance": label.provenance or {},
        "history": label.history or [],
    }


async def collect_pair_metadata(db: Any) -> list[dict[str, Any]]:
    ca, cb = aliased(Claim), aliased(Claim)
    aa, ab = aliased(Article), aliased(Article)
    statement = (
        select(
            EvalPair.id.label("pair_id"), EvalPair.claim_a_id, EvalPair.claim_b_id,
            EvalPair.canonical_key, EvalPair.split, EvalPair.event_group,
            EvalPair.bucket, EvalPair.sim_score.label("cosine_score"), EvalPair.sampled_at,
            EvalPair.sampling_provenance,
            ca.claim_text.label("text_a"), cb.claim_text.label("text_b"),
            ca.article_id.label("article_a_id"), cb.article_id.label("article_b_id"),
            aa.canonical_article_id.label("canonical_article_a_id"),
            ab.canonical_article_id.label("canonical_article_b_id"),
            aa.content_hash.label("article_content_hash_a"), ab.content_hash.label("article_content_hash_b"),
            aa.domain.label("domain_a"), ab.domain.label("domain_b"),
            aa.published_at.label("published_at_a"), ab.published_at.label("published_at_b"),
            aa.url.label("article_url_a"), ab.url.label("article_url_b"),
        )
        .outerjoin(ca, ca.id == EvalPair.claim_a_id)
        .outerjoin(cb, cb.id == EvalPair.claim_b_id)
        .outerjoin(aa, aa.id == ca.article_id)
        .outerjoin(ab, ab.id == cb.article_id)
        .order_by(EvalPair.id)
    )
    output = []
    for row in (await db.execute(statement)).all():
        record = dict(row._mapping)
        for key in ("pair_id", "claim_a_id", "claim_b_id", "article_a_id", "article_b_id", "canonical_article_a_id", "canonical_article_b_id"):
            record[key] = str(record[key]) if record[key] is not None else None
        for key in ("sampled_at", "published_at_a", "published_at_b"):
            record[key] = iso(record[key])
        record["event_id"] = record["event_group"]
        output.append(record)
    return output


async def collect_export_records(db: Any, dataset_version: str) -> list[dict[str, Any]]:
    """Use snapshot_session for a repeatable-read snapshot across these queries."""
    pairs = await collect_pair_metadata(db)
    labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in (await db.execute(select(EvalPairLabel).order_by(EvalPairLabel.pair_id, EvalPairLabel.origin, EvalPairLabel.annotator, EvalPairLabel.id))).scalars():
        labels[str(vote.pair_id)].append(vote_record(vote))
    entities: dict[str, list[str]] = defaultdict(list)
    for claim_id, entity_text in (await db.execute(select(ClaimEntity.claim_id, ClaimEntity.entity_text).order_by(ClaimEntity.claim_id, ClaimEntity.entity_text))).all():
        entities[str(claim_id)].append(entity_text)
    records = []
    for pair in pairs:
        votes = labels.get(str(pair["pair_id"]), [])
        result = pair_consensus(votes)
        pair.update({
            "schema_version": SCHEMA_VERSION,
            "dataset_version": dataset_version,
            "score_type": "cosine_score_not_probability",
            "features": {
                "lexical_jaccard": lexical_jaccard(pair["text_a"], pair["text_b"]) if pair["text_a"] is not None and pair["text_b"] is not None else None,
                "entity_jaccard": entity_jaccard(entities[pair["claim_a_id"]], entities[pair["claim_b_id"]]),
            },
            "feature_provenance": {
                "version": FEATURE_VERSION,
                "lexical_jaccard": "casefolded word-set Jaccard; texts only",
                "entity_jaccard": "stored claim_entities, casefolded entity-set Jaccard; null means no entity evidence",
            },
            "gold_label": result["label"],
            "gold_mutation_types": result["mutation_types"],
            "gold_context_relation": result["context_relation"],
            "gold_changed_spans": result["changed_spans"],
            "consensus": result,
            "human_labels": [v for v in votes if human_identity(v) is not None],
            "labels": votes,
            "provenance": {
                "gold_origin": "human" if result["status"] == "agreed" else None,
                "consensus_rule": CONSENSUS_RULE,
                "origin_counts": origin_counts(votes),
                "score_source": "eval_pairs.sim_score (stored cosine; embedding version not recorded in historical data)",
                "historical_duplicate": pair["canonical_key"] is None,
            },
        })
        records.append(pair)
    return records


@asynccontextmanager
async def snapshot_session() -> AsyncIterator[Any]:
    from app.db.postgres import async_session_maker

    async with async_session_maker() as db:
        if db.get_bind().dialect.name == "postgresql":
            await db.connection(execution_options={"isolation_level": "REPEATABLE READ"})
            await db.execute(text("SET TRANSACTION READ ONLY"))
        try:
            yield db
        finally:
            await db.rollback()


async def lock_pair_writes(db: Any) -> None:
    if db.get_bind().dialect.name == "postgresql":
        # Split assignment and sampling use the same lock, avoiding partial freezes.
        await db.execute(text("LOCK TABLE eval_pairs IN SHARE ROW EXCLUSIVE MODE"))


def _identifier(value: Any, field: str, pid: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise DatasetError(f"Pair {pid}: {field} must be an explicit nonempty string ID; export claim/article provenance before evaluating")
    return value


def _text_key(value: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def resource_tokens(record: dict[str, Any]) -> set[str]:
    tokens = set()
    event = record.get("event_id", record.get("event_group"))
    if event and event != "unassigned":
        tokens.add(f"event:{event}")
    for side in ("a", "b"):
        for kind, key in (("claim", f"claim_{side}_id"), ("article", f"article_{side}_id"), ("article", f"canonical_article_{side}_id"), ("article_hash", f"article_content_hash_{side}")):
            if record.get(key):
                tokens.add(f"{kind}:{record[key]}")
        if record.get(f"text_{side}"):
            tokens.add(f"claim_text:{_text_key(record[f'text_{side}'])}")
    return tokens


def connected_components(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parent = list(range(len(records)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for i, record in enumerate(records):
        for token in sorted(resource_tokens(record)):
            if token in owner:
                a, b = root(i), root(owner[token])
                parent[max(a, b)] = min(a, b)
            else:
                owner[token] = i
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, record in enumerate(records):
        groups[root(i)].append(record)
    return [sorted(group, key=lambda r: str(r["pair_id"])) for _, group in sorted(groups.items())]


def validate_record(record: dict[str, Any]) -> None:
    pid = str(record.get("pair_id", "<missing>"))
    if record.get("schema_version") != SCHEMA_VERSION:
        raise DatasetError(f"Pair {pid}: schema_version must be {SCHEMA_VERSION!r}; use export_research_dataset.py")
    version = record.get("dataset_version")
    if not isinstance(version, str) or not version.strip() or len(version) > 128:
        raise DatasetError(f"Pair {pid}: supply a nonempty dataset_version (at most 128 characters)")
    for field in ("pair_id", "event_id", "claim_a_id", "claim_b_id", "article_a_id", "article_b_id"):
        _identifier(record.get(field), field, pid)
    if record["event_id"].casefold() == "unassigned" or record.get("split") not in SPLITS:
        raise DatasetError(f"Pair {pid}: event_id/split is unassigned; freeze event groups into train/dev/test using POST /eval/splits")
    if record["claim_a_id"] == record["claim_b_id"]:
        raise DatasetError(f"Pair {pid}: a claim cannot be paired with itself")
    for key in ("text_a", "text_b"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise DatasetError(f"Pair {pid}: {key} must contain the actual claim text")
    score = record.get("cosine_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not -1 <= score <= 1:
        raise DatasetError(f"Pair {pid}: cosine_score must be a finite stored cosine in [-1, 1], not a probability")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or record.get("gold_label") not in LABELS or provenance.get("gold_origin") != "human":
        raise DatasetError(f"Pair {pid}: only authenticated human gold is accepted; automatic/legacy/test pseudo-gold is exploratory")
    votes = record.get("human_labels")
    if not isinstance(votes, list) or not votes or any(not isinstance(v, dict) or human_identity(v) is None for v in votes):
        raise DatasetError(f"Pair {pid}: human_labels must include authenticated origin=human, annotator=user:<uuid>, annotator_user_id evidence")
    for vote in votes:
        if str(vote.get("pair_id")) != pid:
            raise DatasetError(f"Pair {pid}: human vote evidence belongs to a different pair; export actual per-pair records")
        _identifier(vote.get("id"), "human vote id", pid)
        vote_mutations = vote.get("mutation_types")
        if vote_mutations is not None and (not isinstance(vote_mutations, list) or len(vote_mutations) > 16 or any(not isinstance(t, str) or not t.strip() or len(t) > 64 for t in vote_mutations)):
            raise DatasetError(f"Pair {pid}: human mutation annotations must be null or a bounded string list")
    result = pair_consensus(votes)
    if result["status"] != "agreed" or result["label"] != record["gold_label"]:
        raise DatasetError(f"Pair {pid}: gold requires at least two distinct authenticated humans with no disagreement; got {result['status']}")
    mutations = record.get("gold_mutation_types")
    if mutations is not None:
        if not isinstance(mutations, list) or len(mutations) > 16 or any(not isinstance(t, str) or not t.strip() or len(t) > 64 for t in mutations):
            raise DatasetError(f"Pair {pid}: gold_mutation_types must be null or at most 16 bounded strings")
        if result["mutation_status"] != "agreed" or sorted(set(mutations)) != result["mutation_types"]:
            raise DatasetError(f"Pair {pid}: mutation gold also requires independently agreeing human mutation annotations; omit unresolved mutation gold")
    for field in ("context_relation", "changed_spans"):
        value = record.get(f"gold_{field}")
        if value is None:
            continue
        if result[f"{field}_status"] != "agreed" or value != result[field]:
            raise DatasetError(f"Pair {pid}: {field} gold requires independently agreeing human annotations under the same rubric")
        try:
            if field == "context_relation":
                TypeAdapter(ContextRelation).validate_python(value)
            else:
                if not isinstance(value, list) or len(value) > 32:
                    raise ValueError("expected at most 32 changed spans")
                validate_spans([ChangedSpan.model_validate(span) for span in value], record["text_a"], record["text_b"])
        except (ValidationError, ValueError) as exc:
            raise DatasetError(f"Pair {pid}: invalid {field} annotation: {exc}") from exc
    features = record.get("features", {})
    if not isinstance(features, dict) or set(features) - {"lexical_jaccard", "entity_jaccard"}:
        raise DatasetError(f"Pair {pid}: supported optional features are lexical_jaccard and entity_jaccard only")
    for key, value in features.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
            raise DatasetError(f"Pair {pid}: feature {key} must be null or a finite number in [0, 1]")


def validate_dataset(records: list[dict[str, Any]], require_support: bool = True) -> dict[str, Any]:
    if not records:
        raise DatasetError("No agreed human-gold records; obtain at least two independent authenticated human labels per pair, then export")
    seen_ids: set[str] = set()
    seen_pairs: dict[tuple[str, str], str] = {}
    seen_vote_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise DatasetError("Every JSONL line must be a record object")
        validate_record(record)
        pid = record["pair_id"]
        key = tuple(sorted((record["claim_a_id"], record["claim_b_id"])))
        if pid in seen_ids or key in seen_pairs:
            previous = seen_pairs.get(key, pid)
            raise DatasetError(f"Duplicate/reversed pair {pid} (representative {previous}); historical votes are retained, but curate one representative in a new snapshot without deleting DB history")
        seen_ids.add(pid)
        seen_pairs[key] = pid
        for vote in record["human_labels"]:
            if vote["id"] in seen_vote_ids:
                raise DatasetError(f"Pair {pid}: repeated human vote id {vote['id']}; independent per-pair evidence cannot be copied or counted twice")
            seen_vote_ids.add(vote["id"])
    versions = sorted({r["dataset_version"] for r in records})
    if len(versions) != 1:
        raise DatasetError(f"Mixed dataset versions {versions}; evaluate one immutable version at a time")
    components = connected_components(records)
    group_by_pair = {}
    groups_by_split: Counter[str] = Counter()
    for component in components:
        splits = sorted({r["split"] for r in component})
        if len(splits) != 1:
            pairs = [r["pair_id"] for r in component]
            events = sorted({r["event_id"] for r in component})
            raise DatasetError(f"Leakage: connected event/claim/article/identical-text group spans {splits}; events={events}, pairs={pairs[:20]}. Merge the whole connected component into one split before fitting")
        group_id = "group:" + hashlib.sha256("\n".join(r["pair_id"] for r in component).encode()).hexdigest()[:24]
        group_by_pair.update({r["pair_id"]: group_id for r in component})
        groups_by_split[splits[0]] += 1
    support = {}
    for split in SPLITS:
        partition = [r for r in records if r["split"] == split]
        positives = sum(r["gold_label"] in POSITIVE_LABELS for r in partition)
        support[split] = {
            "pairs": len(partition), "positive": positives, "negative": len(partition) - positives,
            "events": len({r["event_id"] for r in partition}), "independent_groups": groups_by_split[split],
        }
        if require_support and (positives < 2 or len(partition) - positives < 2 or groups_by_split[split] < 2):
            raise DatasetError(f"Insufficient {split}: {support[split]}. Need at least two positive pairs, two negative pairs and two independent connected event groups in each split; collect more human gold, not machine votes. This is a computability floor, not a publication guarantee")
    return {"dataset_version": versions[0], "support": support, "group_by_pair": group_by_pair, "leakage_checked": ["event_id", "claim_ids", "article_ids", "canonical_article_ids", "article_content_hashes", "normalized_identical_claim_texts"]}


def validate_assignments(records: list[dict[str, Any]], changes: list[dict[str, Any]]) -> None:
    by_id = {str(r["pair_id"]): r for r in records}
    changed_ids = set()
    for change in changes:
        pid = str(change["pair_id"])
        if pid in changed_ids:
            raise DatasetError(f"Duplicate split assignment for {pid}")
        changed_ids.add(pid)
        if pid not in by_id:
            raise DatasetError(f"Pair {pid} not found")
        record = by_id[pid]
        old_event = record.get("event_id", record.get("event_group", "unassigned"))
        if record["split"] != "unassigned" and (record["split"] != change["split"] or old_event != change["event_group"]):
            raise DatasetError(f"Pair {pid} is frozen as {old_event}/{record['split']}; create a new external dataset version rather than moving held-out pairs")
        record.update({"split": change["split"], "event_id": change["event_group"], "event_group": change["event_group"]})
    for component in connected_components(records):
        if not any(r["pair_id"] in changed_ids for r in component):
            continue
        splits = {r["split"] for r in component}
        events = {r["event_id"] for r in component}
        if "unassigned" in splits or "unassigned" in events:
            missing = [r["pair_id"] for r in component if r["split"] == "unassigned" or r["event_id"] == "unassigned"]
            raise DatasetError(f"Assign the entire connected event/claim/article component together; missing pair_ids={missing[:50]}")
        if len(splits) != 1 or len(events) != 1:
            raise DatasetError(f"Connected pairs cannot cross event groups or splits: pair_ids={[r['pair_id'] for r in component][:50]}; assign one event_group/split for the whole component")


def build_export(records: list[dict[str, Any]], dataset_version: str, publication: bool = False, allow_incomplete: bool = False) -> dict[str, Any]:
    if not dataset_version.strip() or len(dataset_version) > 128:
        raise DatasetError("Supply a dataset_version of 1–128 characters")
    exclusions: Counter[str] = Counter()
    selected = []
    unassigned = []
    for record in records:
        status = record["consensus"]["status"]
        if status != "agreed":
            exclusions[status] += 1
        elif record["split"] not in SPLITS or record["event_id"] == "unassigned":
            exclusions["unassigned_event_or_split"] += 1
            unassigned.append(record["pair_id"])
        else:
            selected.append(record)
    validation = None
    if publication:
        if unassigned and not allow_incomplete:
            raise DatasetError(f"Human-gold pairs have unassigned event/split: {unassigned[:50]}; freeze them using POST /eval/splits, or explicitly use --allow-incomplete to exclude and report them")
        validation = validate_dataset(selected, require_support=False)
    exported = selected if publication else records
    warnings = [
        "Export is a snapshot, not a publication-validity assurance. Review event definitions, sampling bias and annotation independence; use the train/dev/test research runner.",
        "Historical automatic/legacy/test votes and revisions are retained as provenance, never used as primary gold.",
        "Stored cosine is a score, not a calibrated probability. Historical embedding model/version is unknown unless sampling provenance records it.",
        "Claim/article text, annotator UUIDs and notes may require licensing/consent before distributing this snapshot.",
    ]
    if unassigned:
        warnings.append(f"{len(unassigned)} human-gold pairs have unassigned events/splits" + (" and were explicitly excluded" if publication else " and cannot enter a research experiment"))
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset_version": dataset_version,
        "record_count": len(exported), "source_pair_count": len(records),
        "human_gold_assigned_count": len(selected), "exclusions": dict(sorted(exclusions.items())),
        "publication_mode": publication, "publication_ready": False,
        "records_sha256": hashlib.sha256(records_bytes(exported)).hexdigest(),
        "consensus_rule": CONSENSUS_RULE,
        "label_origins": {origin: sum(r["provenance"]["origin_counts"][origin]["votes"] for r in records) for origin in ORIGINS},
        "validation": validation, "warnings": warnings,
    }
    return {"schema_version": SCHEMA_VERSION, "dataset_version": dataset_version, "manifest": manifest, "records": exported}
