# mypy: disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-any-generics=False

"""Exploratory weak-label generation, never authenticated human gold.

Preserves the existing NLI, entity-overlap and lexical signals, but these
machine voters are not independent humans. Entity/lexical methods retain the
sim_score argument for compatibility and deliberately ignore it, removing
circular dependence on the engine score being evaluated. Failures leave a
signal unlabeled instead of inventing a DISTINCT vote.

Usage: python scripts/auto_label_eval_pairs.py [--batch-size 20] [--dry-run]

Writes origin=automatic explicitly, skips only existing automatic votes,
and never deletes, upgrades, or overwrites historical legacy/human votes.
Model loading may download FLAN-T5/spaCy resources; weak labels remain
exploratory regardless of agreement or sample size.
"""

import argparse
import asyncio
import logging
import re
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from spacy.language import Language

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.postgres import async_session_maker
from app.models.eval import EvalPairLabel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

ANNOTATOR_NLI = "nli-annotator"
ANNOTATOR_ENTITY = "entity-annotator"
ANNOTATOR_LEXICAL = "lexical-annotator"
ALL_ANNOTATORS = {ANNOTATOR_NLI, ANNOTATOR_ENTITY, ANNOTATOR_LEXICAL}


# ─────────────────────────────────────────────────────────────────────────────
# Signal 1: NLI stance (FLAN-T5-base)
# ─────────────────────────────────────────────────────────────────────────────

_nli_tok: Any = None
_nli_mdl: Any = None


def _load_nli() -> tuple[Any, Any]:
    global _nli_tok, _nli_mdl
    if _nli_mdl is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading NLI model google/flan-t5-base...")
        _nli_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
        _nli_mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        _nli_mdl.eval()
    return _nli_tok, _nli_mdl


def nli_label(text_a: str, text_b: str) -> str:
    """Entailment-based labeling: check BOTH directions.

    If A entails B AND B entails A → SAME_STORY (mutual entailment).
    If A entails B OR B entails A → EVOLVED (asymmetric).
    If neither → DISTINCT.
    If either contradicts → DISTINCT.
    """
    import torch

    tok, mdl = _load_nli()

    def _check(premise: str, hypothesis: str) -> str:
        prompt = (
            f"premise: {premise[:300]} hypothesis: {hypothesis[:300]} "
            f"Does the premise entail the hypothesis?"
        )
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=3, do_sample=False)
        ans = tok.batch_decode(out, skip_special_tokens=True)[0].strip().lower()
        return "yes" if ans == "yes" else ("no" if ans == "no" else "neutral")

    # Check both directions
    ab = _check(text_a, text_b)
    ba = _check(text_b, text_a)

    if ab == "no" or ba == "no":
        return "DISTINCT"
    if ab == "yes" and ba == "yes":
        return "SAME_STORY"
    if ab == "yes" or ba == "yes":
        return "EVOLVED"
    # Both neutral
    return "DISTINCT"


# ─────────────────────────────────────────────────────────────────────────────
# Signal 2: Named-entity overlap (spaCy)
# ─────────────────────────────────────────────────────────────────────────────

_spacy_nlp: Language | None = None


def _load_spacy() -> Language:
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        logger.info("Loading spaCy en_core_web_sm...")
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def entity_label(text_a: str, text_b: str, sim_score: float) -> str:
    """Text-only weak entity overlap; deprecated sim_score is ignored."""
    nlp = _load_spacy()
    doc_a = nlp(text_a[:500])
    doc_b = nlp(text_b[:500])

    # Extract unique entity strings (lowercased for fuzzy matching)
    ents_a = {ent.text.lower().strip() for ent in doc_a.ents if len(ent.text.strip()) > 1}
    ents_b = {ent.text.lower().strip() for ent in doc_b.ents if len(ent.text.strip()) > 1}

    if not ents_a or not ents_b:
        return "DISTINCT"

    intersection = ents_a & ents_b
    union = ents_a | ents_b
    jaccard = len(intersection) / len(union) if union else 0.0

    # Also check partial matches (substring containment)
    partial_matches = 0
    for ea in ents_a:
        for eb in ents_b:
            if ea != eb and (ea in eb or eb in ea):
                partial_matches += 1

    effective_overlap = jaccard + 0.1 * min(partial_matches, 3)

    if effective_overlap >= 0.5:
        return "SAME_STORY"
    if effective_overlap >= 0.3:
        return "EVOLVED"
    return "DISTINCT"


# ─────────────────────────────────────────────────────────────────────────────
# Signal 3: Lexical overlap (TF-IDF + word Jaccard)
# ─────────────────────────────────────────────────────────────────────────────

_tfidf_vec = None


def _tokenize(text_str: str) -> set[str]:
    """Simple word tokenization with lowercasing."""
    return set(re.findall(r"\b[a-z][a-z0-9'-]{1,}\b", text_str.lower()))


def lexical_label(text_a: str, text_b: str, sim_score: float) -> str:
    """Text-only word/char weak signal; deprecated sim_score is ignored."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Word-level Jaccard
    words_a = _tokenize(text_a)
    words_b = _tokenize(text_b)

    if not words_a or not words_b:
        return "DISTINCT"

    word_jaccard = len(words_a & words_b) / len(words_a | words_b)

    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
    )
    mat = vec.fit_transform([text_a, text_b])
    tfidf_cos = float(cosine_similarity(mat[0:1], mat[1:2])[0, 0])

    # Combined lexical score (weighted average)
    lexical_score = 0.4 * word_jaccard + 0.6 * tfidf_cos

    # Heuristic boundaries, not a calibration or an assertion of label quality.
    if lexical_score >= 0.40:
        return "SAME_STORY"
    if lexical_score >= 0.20:
        return "EVOLVED"
    return "DISTINCT"


# ─────────────────────────────────────────────────────────────────────────────
# Main labeling pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_unlabeled_pairs(batch_size: int | None = None) -> list[dict]:
    """Fetch eval pairs that are missing labels from any of the 3 annotators."""
    async with async_session_maker() as db:
        rows = (
            await db.execute(
                text("""
                    SELECT p.id::text AS pair_id,
                           p.bucket,
                           p.sim_score,
                           ca.claim_text AS text_a,
                           cb.claim_text AS text_b,
                           (SELECT array_agg(l.annotator)
                            FROM eval_pair_labels l
                            WHERE l.pair_id = p.id AND l.origin = 'automatic') AS existing_annotators
                    FROM eval_pairs p
                    JOIN claims ca ON ca.id = p.claim_a_id
                    JOIN claims cb ON cb.id = p.claim_b_id
                    ORDER BY p.bucket, p.sim_score DESC
                """)
            )
        ).all()

    pairs = []
    for r in rows:
        existing = set(r[5] or [])
        missing = ALL_ANNOTATORS - existing
        if missing:
            pairs.append({
                "pair_id": r[0],
                "bucket": r[1],
                "sim_score": float(r[2]),
                "text_a": r[3],
                "text_b": r[4],
                "missing_annotators": missing,
            })

    if batch_size:
        pairs = pairs[:batch_size]
    return pairs


async def save_labels(labels: list[dict]) -> int:
    """Race-safe automatic-only inserts; all existing votes remain untouched."""
    if not labels:
        return 0
    async with async_session_maker() as db:
        inserted = 0
        for lab in labels:
            if lab["annotator"] not in ALL_ANNOTATORS or lab["label"] not in {"SAME_STORY", "EVOLVED", "DISTINCT"}:
                raise ValueError("Only known machine signals with valid labels can be saved automatically")
            statement = pg_insert(EvalPairLabel).values(
                id=uuid.uuid4(), pair_id=uuid.UUID(str(lab["pair_id"])),
                annotator=lab["annotator"], label=lab["label"], origin="automatic",
                provenance={"method": lab["annotator"], "version": "weak-labels-v2-no-engine-cosine", "inputs": ["text_a", "text_b"], "independent_human": False},
            ).on_conflict_do_nothing(constraint="uq_eval_label_pair_annotator_origin").returning(EvalPairLabel.id)
            inserted += int((await db.execute(statement)).scalar_one_or_none() is not None)
        await db.commit()
    return inserted


def label_pair(pair: dict) -> list[dict]:
    """Generate labels for one pair from all missing annotators."""
    results = []
    text_a = pair["text_a"]
    text_b = pair["text_b"]
    sim = pair["sim_score"]

    for annotator in sorted(pair["missing_annotators"]):
        try:
            if annotator == ANNOTATOR_NLI:
                label = nli_label(text_a, text_b)
            elif annotator == ANNOTATOR_ENTITY:
                label = entity_label(text_a, text_b, sim)
            elif annotator == ANNOTATOR_LEXICAL:
                label = lexical_label(text_a, text_b, sim)
            else:
                continue

            results.append({
                "pair_id": pair["pair_id"],
                "annotator": annotator,
                "origin": "automatic",
                "label": label,
            })
        except Exception as e:
            logger.warning("Signal failed; leaving pair %s unlabeled by %s: %s", pair["pair_id"], annotator, e)

    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exploratory automatic weak labels (not human gold)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Max pairs to process (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print labels without saving")
    args = parser.parse_args()

    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    logger.warning("EXPLORATORY WEAK LABELS: machine signals are not independent human annotators or publication gold")
    logger.info("Fetching pairs missing automatic signals...")
    pairs = await fetch_unlabeled_pairs(args.batch_size)
    logger.info("Found %d pairs needing labels", len(pairs))

    if not pairs:
        logger.info("All pairs already have the requested automatic signals; human coverage is unchanged")
        return

    # Preload models once
    logger.info("Preloading NLP models...")
    _load_nli()
    _load_spacy()

    all_labels: list[dict] = []
    batch_labels: list[dict] = []
    stats: dict[str, Counter] = {
        ANNOTATOR_NLI: Counter(),
        ANNOTATOR_ENTITY: Counter(),
        ANNOTATOR_LEXICAL: Counter(),
    }

    for i, pair in enumerate(pairs):
        labels = label_pair(pair)
        all_labels.extend(labels)
        batch_labels.extend(labels)
        for lab in labels:
            stats[lab["annotator"]][lab["label"]] += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
            if not args.dry_run and batch_labels:
                await save_labels(batch_labels)
                batch_labels.clear()
            
            logger.info(
                "Progress: %d / %d pairs labeled (%d labels generated)",
                i + 1, len(pairs), len(all_labels),
            )

    # Print distribution
    logger.info("=" * 60)
    logger.info("LABEL DISTRIBUTION")
    logger.info("=" * 60)
    for annotator, counts in stats.items():
        total = sum(counts.values())
        logger.info("  %s (%d labels):", annotator, total)
        for label in ("SAME_STORY", "EVOLVED", "DISTINCT"):
            n = counts.get(label, 0)
            pct = (n / total * 100) if total else 0
            logger.info("    %-16s %4d (%5.1f%%)", label, n, pct)

    # Majority consensus preview
    pair_labels: dict[str, list[str]] = {}
    for lab in all_labels:
        pair_labels.setdefault(lab["pair_id"], []).append(lab["label"])
    consensus_dist: Counter[str] = Counter()
    for labs in pair_labels.values():
        counts = Counter(labs)
        top_count = max(counts.values())
        winners = sorted(l for l, c in counts.items() if c == top_count)
        winner = winners[0] if len(winners) == 1 else "UNRESOLVED"
        consensus_dist[winner] += 1

    logger.info("\n  WEAK-LABEL BATCH MAJORITY PREVIEW (not human gold):")
    total_consensus = sum(consensus_dist.values())
    for label in ("SAME_STORY", "EVOLVED", "DISTINCT"):
        n = consensus_dist.get(label, 0)
        pct = (n / total_consensus * 100) if total_consensus else 0
        logger.info("    %-16s %4d (%5.1f%%)", label, n, pct)

    positive = consensus_dist.get("SAME_STORY", 0) + consensus_dist.get("EVOLVED", 0)
    logger.info("  Positive rate (SAME_STORY + EVOLVED): %.1f%%",
                (positive / total_consensus * 100) if total_consensus else 0)

    if args.dry_run:
        logger.info("DRY RUN — no labels saved")
        return

    # Skip the final save since we saved incrementally in batches
    logger.info("Automatic weak-label generation complete; historical votes were retained and no human gold was created.")


if __name__ == "__main__":
    asyncio.run(main())
