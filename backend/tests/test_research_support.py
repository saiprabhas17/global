"""Synthetic unit-test evidence only; not a real human dataset or research result."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.article import Article
from app.models.claim import Claim
from app.models.eval import EvalPair
from app.models.source import Source, SourceTypeEnum
from app.models.user import RoleEnum, User
from app.services.eval.dataset import SCHEMA_VERSION
from app.services.eval.provenance import origin_counts, pair_consensus


def fixture_user_id(index: int) -> uuid.UUID:
    # PostgreSQL UUID declarations have numeric affinity in SQLite: a hex UUID
    # like ...03e9 can be interpreted as scientific notation instead of text.
    return uuid.uuid5(uuid.NAMESPACE_URL, f"gmee-synthetic-eval-user:{index}")


def human_vote(pid: str, label: str, user: int = 1, mutations: list[str] | None = None) -> dict[str, Any]:
    user_id = str(fixture_user_id(user))
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"synthetic-vote:{pid}:{user}")),
        "pair_id": str(pid), "annotator": f"user:{user_id}", "annotator_user_id": user_id,
        "origin": "human", "label": label, "mutation_types": mutations, "notes": None,
        "provenance": {"method": "synthetic-unit-test-only"}, "history": [],
    }


def research_records(mutations: bool = False, groups: int = 3) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split_index, split in enumerate(("train", "dev", "test")):
        for group in range(groups):
            event = f"{split}-event-{group}"
            for positive in (True, False):
                pid = str(uuid.UUID(int=1 + len(records)))
                label = "EVOLVED" if positive and group == 1 else "SAME_STORY" if positive else "DISTINCT"
                gold_types = (["NUMERIC_DRIFT"] if positive else []) if mutations else None
                votes = [human_vote(pid, label, user=user, mutations=gold_types) for user in (1, 2)]
                text_a = f"Atlas {event} case {pid} announced 10 new instruments for the research mission."
                text_b = f"Atlas {event} case {pid} announced 12 new instruments for the research mission." if positive else f"Birch {event} case {pid} discussed distant rainforests and marine ecosystems."
                cosine = (0.82 - group * 0.07) if positive else (0.18 + group * 0.07)
                if split_index == 2 and group == 2:
                    cosine = 0.42 if positive else 0.59
                record = {
                    "schema_version": SCHEMA_VERSION, "dataset_version": "synthetic-unit-test-v1",
                    "pair_id": pid, "event_id": event, "event_group": event, "split": split,
                    "claim_a_id": f"claim:{pid}:a", "claim_b_id": f"claim:{pid}:b",
                    "article_a_id": f"article:{pid}:a", "article_b_id": f"article:{pid}:b",
                    "canonical_article_a_id": None, "canonical_article_b_id": None,
                    "article_content_hash_a": f"hash:{pid}:a", "article_content_hash_b": f"hash:{pid}:b",
                    "text_a": text_a, "text_b": text_b, "cosine_score": cosine,
                    "features": {"lexical_jaccard": 0.8 if positive else 0.1, "entity_jaccard": (0.7 if positive else 0.05)},
                    "feature_provenance": {"version": "synthetic-manually-specified-test-features"},
                    "gold_label": label, "gold_mutation_types": gold_types,
                    "consensus": pair_consensus(votes), "human_labels": votes, "labels": votes,
                    "bucket": "b75" if cosine >= 0.75 else "b40", "canonical_key": f"claim:{pid}:a:claim:{pid}:b",
                    "sampling_provenance": {"method": "synthetic-unit-test-only"},
                    "provenance": {"gold_origin": "human", "origin_counts": origin_counts(votes), "source": "synthetic-unit-test-only"},
                }
                records.append(record)
    return records


async def seed_user(db: AsyncSession, index: int = 1, role: RoleEnum = RoleEnum.user) -> User:
    user_id = fixture_user_id(index)
    user = await db.get(User, user_id)
    if user is None:
        user = User(id=user_id, email=f"eval-{index}@example.com", hashed_password="synthetic-fixture-not-a-login-password", role=role, is_active=True)
        db.add(user)
        await db.flush()
    return user


async def seed_pair(
    db: AsyncSession, *, sim: float = 0.75, split: str = "unassigned",
    event: str = "unassigned", claim_a: Claim | None = None, claim_b: Claim | None = None,
    pair_id: uuid.UUID | None = None, sampled_at: datetime | None = None,
    text_a: str | None = None, text_b: str | None = None,
) -> tuple[EvalPair, Claim, Claim]:
    async def make_claim(side: str, supplied_text: str | None) -> Claim:
        source = Source(name=f"Synthetic {side}", type=SourceTypeEnum.rss, url_or_identifier=f"https://{side}.example/feed")
        db.add(source)
        await db.flush()
        article_id = uuid.uuid4()
        article = Article(id=article_id, source_id=source.id, title=f"Synthetic article {side}", url=f"https://{side}.example/{article_id}", content_hash=str(article_id), domain=f"{side}.example", published_at=datetime(2026, 9, 1, tzinfo=UTC))
        db.add(article)
        await db.flush()
        claim_id = uuid.uuid4()
        claim = Claim(id=claim_id, article_id=article.id, claim_text=supplied_text or f"Synthetic claim {side} {claim_id} reports an event for annotation.")
        db.add(claim)
        await db.flush()
        return claim

    claim_a = claim_a or await make_claim("a", text_a)
    claim_b = claim_b or await make_claim("b", text_b)
    pair = EvalPair(id=pair_id or uuid.uuid4(), claim_a_id=claim_a.id, claim_b_id=claim_b.id, bucket="b75", sim_score=sim, split=split, event_group=event, sampled_at=sampled_at or datetime(2026, 9, 2, tzinfo=UTC))
    db.add(pair)
    await db.flush()
    return pair, claim_a, claim_b
