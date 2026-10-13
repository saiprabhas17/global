"""Evidence-gated verdict bands and local three-class textual stance.

The legacy ``probability`` field is retained for storage/API compatibility,
not as a calibrated probability of truth. It is a hand-weighted heuristic.
Similarity retrieves candidates; only explicit NLI entailment can establish
corroboration, and only explicit contradiction can establish dispute.
Language, entities and source priors cannot establish support on their own.

Source priors default to neutral. Supplied priors must disclose independently
validated data and provenance, never counts of this engine's own verdicts.
Corpus agreement and different domains do not prove factual truth or source
independence. New evidence records disclose these limitations and a version;
previously stored verdicts are neither deleted nor automatically rescored.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any, Literal, TypedDict
from urllib.parse import urlsplit

from app.core.config import get_settings

logger = logging.getLogger(__name__)

METHOD_VERSION = "corpus-nli-v1"
SCORE_KIND: Literal["uncalibrated_heuristic"] = "uncalibrated_heuristic"
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
Stance = Literal["entailment", "contradiction", "neutral"]
LegacyStance = Literal["yes", "no", "neutral"]

# Hand-set weights, not fitted or calibrated probabilities.
W_CORROB = 0.40
W_CONTRA = 0.20
W_TRACK = 0.15
W_ENTITY = 0.15
W_LANGUAGE = 0.10

# The configured similarity window is retained for legacy introspection.
# Corpus checking uses its minimum for retrieval, not for deciding stance;
# exact matches remain eligible and are deduplicated by article provenance.


def near_window() -> tuple[float, float]:
    """Effective (min, max) similarity window, settings-overridable."""
    s = get_settings()
    return (s.VERDICT_NEAR_MIN, s.VERDICT_NEAR_MAX)


class BandSpec(TypedDict, total=False):
    """One advertised verdict band. `when` documents override conditions."""

    threshold: float
    band: str
    when: str


class NLIConfiguration(TypedDict):
    model: str
    revision: str | None
    confidence_min: float


def nli_configuration() -> NLIConfiguration:
    """Inspect configuration without importing or loading a model."""
    settings = get_settings()
    model = getattr(settings, "NLI_MODEL", DEFAULT_NLI_MODEL)
    revision = getattr(settings, "NLI_MODEL_REVISION", None)
    confidence_min = float(getattr(settings, "NLI_CONFIDENCE_MIN", 0.7))
    if not isinstance(model, str) or not model.strip():
        raise ValueError("NLI_MODEL must name a local model or cached model ID")
    if not math.isfinite(confidence_min) or not 0 <= confidence_min <= 1:
        raise ValueError("NLI_CONFIDENCE_MIN must be between 0 and 1")
    return {
        "model": model.strip(),
        "revision": str(revision) if revision else None,
        "confidence_min": confidence_min,
    }


class EngineConfig(TypedDict):
    """Machine-readable shape of `engine_config()` (the XAI contract)."""

    weights: dict[str, float]
    bands: list[BandSpec]
    near_similarity_window: dict[str, float]
    similarity_role: str
    corroboration_rule: str
    laplace_alpha_smoothing: float
    pooling: str
    disputed_rule: str
    source_prior_policy: str
    score_kind: str
    method_version: str
    nli: NLIConfiguration
    stance_cache: StanceCacheConfig
    warnings: list[str]


class StanceCacheConfig(TypedDict):
    key: str
    max_entries: int
    ttl_seconds: float


def engine_config() -> EngineConfig:
    """Full introspection of the engine's current configuration.

    Exposed verbatim via GET /verdicts/summary so reviewers can verify that
    the deployed weights match the published methodology — the XAI contract.
    """
    win_min, win_max = near_window()
    bands: list[BandSpec] = [
        BandSpec(
            threshold=t,
            band=b,
            when=(
                "explicit cross-domain entailment and no contradiction"
                if b in SUPPORTED_BANDS or b == "WEAKLY_CORROBORATED"
                else "no established support; absence of evidence is not falsity"
            ),
        )
        for t, b in VERDICT_BANDS
    ]
    bands.append(
        BandSpec(
            threshold=DISPUTED_MAX_PROBABILITY,
            band="DISPUTED",
            when="explicit NLI contradiction and heuristic score below threshold",
        )
    )
    return {
        "weights": {
            "corroboration": W_CORROB,
            "contradiction": W_CONTRA,
            "source_track_record": W_TRACK,
            "entity_grounding": W_ENTITY,
            "language": W_LANGUAGE,
        },
        "bands": bands,
        "near_similarity_window": {
            "min": win_min,
            "max": win_max,
        },
        "similarity_role": "retrieval_only; minimum applied, exact matches eligible",
        "corroboration_rule": "explicit entailment only; canonical duplicates and "
        "same-domain repetition are not independent support",
        "laplace_alpha_smoothing": LAPLACE_ALPHA,
        "pooling": "hand-weighted log-odds transformed to an uncalibrated score",
        "disputed_rule": "explicit contradiction and score below threshold; "
        "otherwise contradiction yields UNRESOLVED",
        "source_prior_policy": "neutral unless independently validated counts "
        "and provenance are supplied; never self-verdicts",
        "score_kind": SCORE_KIND,
        "method_version": METHOD_VERSION,
        "nli": nli_configuration(),
        "stance_cache": {
            "key": "SHA-256 of full ordered texts, model, revision and threshold",
            "max_entries": STANCE_CACHE_MAX,
            "ttl_seconds": STANCE_CACHE_TTL_SECONDS,
        },
        "warnings": [
            "Scores are uncalibrated heuristics, not probabilities of truth.",
            "Corpus-local textual agreement is not independent fact verification.",
            "Different domains do not guarantee independent reporting.",
        ],
    }


LAPLACE_ALPHA = 4.0  # smoothing for outlet track record

HEDGE_TERMS = (
    "reportedly",
    "allegedly",
    "rumor",
    "rumour",
    "unconfirmed",
    "sources say",
    "it is said",
    "claims",
    "supposedly",
    "appears to",
)
SENSATION_TERMS = (
    "shocking",
    "bombshell",
    "exposed",
    "conspiracy",
    "secret",
    "they don't want you",
    "wake up",
    "mainstream media",
)

VERDICT_BANDS = (
    (0.72, "SUPPORTED"),  # multiple independent outlets agree
    (0.55, "PARTIALLY_SUPPORTED"),  # some corroboration, thin
    (0.38, "UNRESOLVED"),  # genuinely uncertain
    (0.00, "WEAKLY_CORROBORATED"),  # below 0.38 but neighbours back it
    (0.00, "UNSUPPORTED"),  # below 0.38, single-source / no support
)

# DISPUTED is an override rather than a threshold: band_for() only returns it
# when an explicit NLI contradiction exists AND the probability is this low.
DISPUTED_MAX_PROBABILITY = 0.45

# Legacy band vocabulary retained for callers and stored records.
SUPPORTED_BANDS = ("SUPPORTED", "PARTIALLY_SUPPORTED")
DISPUTED_BANDS = ("DISPUTED",)


def band_for(
    p: float,
    has_contradiction: bool = False,
    has_corroboration: bool = False,
) -> str:
    """Gate legacy score bands on actual stance evidence, not style or cosine."""
    if has_contradiction:
        return "DISPUTED" if p < DISPUTED_MAX_PROBABILITY else "UNRESOLVED"
    if not has_corroboration:
        return "UNRESOLVED" if p >= 0.38 else "UNSUPPORTED"
    if p >= 0.72:
        return "SUPPORTED"
    if p >= 0.55:
        return "PARTIALLY_SUPPORTED"
    if p >= 0.38:
        return "UNRESOLVED"
    return "WEAKLY_CORROBORATED"


def canonical_domain(domain: str | None) -> str | None:
    """Normalize host spelling without claiming registrable-domain independence."""
    if not domain or not domain.strip():
        return None
    host = urlsplit(domain if "://" in domain else f"//{domain}").hostname
    return host.lower().removeprefix("www.").rstrip(".") if host else None


@dataclass
class Signal:
    name: str
    value: float  # normalised 0..1 where 1 pushes toward SUPPORTED
    weight: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerdictResult:
    probability: float  # Compatibility name for an uncalibrated heuristic score.
    band: str
    rationale: str
    evidence: dict[str, Any]


class VerdictEngine:
    """Stateless scorer; corpus statistics are passed in per call so the
    engine itself stays pure and unit-testable."""

    # -------------------------------------------------- language signal

    @staticmethod
    def linguistic_signal(claim_text: str) -> Signal:
        low = claim_text.lower()
        words = max(1, len(low.split()))
        hedges = sum(1 for t in HEDGE_TERMS if t in low)
        sensation = sum(1 for t in SENSATION_TERMS if t in low)

        # base: declarative, specific sentences score higher
        score = 0.85
        score -= 0.12 * hedges
        score -= 0.18 * sensation
        if re.search(r"\b(i|we|my|our|you)\b", low):
            score -= 0.15
        if re.search(r"\d", claim_text):
            score += 0.05
        if words < 6:
            score -= 0.10

        return Signal(
            name="language",
            value=max(0.05, min(1.0, score)),
            weight=W_LANGUAGE,
            detail={"hedges": hedges, "sensational": sensation, "words": words},
        )

    # -------------------------------------------------- entity signal

    @staticmethod
    def entity_signal(entities: list[dict[str, Any]]) -> Signal:
        strong_types = {
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "DATE",
            "CARDINAL",
            "MONEY",
            "QUANTITY",
            "TIME",
            "PERCENT",
        }
        strong = sum(1 for e in entities if e.get("entity_type") in strong_types)
        total = len(entities)

        if total == 0:
            value = 0.30
        elif strong == 0:
            value = 0.45
        else:
            value = min(1.0, 0.55 + 0.15 * strong)

        return Signal(
            name="entity_grounding",
            value=value,
            weight=W_ENTITY,
            detail={"strong_entities": strong, "total_entities": total},
        )

    # -------------------------------------------------- corroboration

    @staticmethod
    def corroboration_signal(
        neighbor_claims: list[dict[str, Any]],
        own_domain: str | None,
        own_syndication_group: str | None = None,
    ) -> tuple[Signal, Signal]:
        """Use canonical stance (or legacy yes/no), never similarity as support."""
        supporting_domains: set[str] = set()
        contradicting_domains: set[str] = set()
        supporting_groups: set[str] = set()
        contradicting_groups: set[str] = set()
        seen_groups: set[str] = set()
        neutral_nearby = 0
        own_domain = canonical_domain(own_domain)

        for nb in neighbor_claims:
            dom = canonical_domain(nb.get("domain"))
            group = nb.get("syndication_group")
            if not dom or dom == own_domain:
                continue
            if group and (group == own_syndication_group or group in seen_groups):
                continue
            if group:
                seen_groups.add(group)
            stance = nb.get("stance", nb.get("nli"))
            if stance in ("contradiction", "no"):
                contradicting_domains.add(dom)
                contradicting_groups.add(group or f"domain:{dom}")
            elif stance in ("entailment", "yes"):
                supporting_domains.add(dom)
                supporting_groups.add(group or f"domain:{dom}")
            else:
                neutral_nearby += 1

        n_sup = len(supporting_domains)
        n_con = len(contradicting_domains)

        # corroboration: saturating curve — 1 independent outlet is already
        # meaningful, 3+ is strong
        corr_value = 0.35 + 0.65 * (1 - math.exp(-n_sup / 1.5)) if n_sup else 0.30
        contra_ratio = n_con / (n_con + n_sup) if (n_con + n_sup) else 0.0

        corroboration = Signal(
            name="corroboration",
            value=round(corr_value, 3),
            weight=W_CORROB,
            detail={
                "independent_outlets_supporting": sorted(supporting_domains),
                "supporting_syndication_groups": sorted(supporting_groups),
                "nearby_unrelated": neutral_nearby,
                "support_rule": "explicit_entailment_only",
                "independence_disclosure": "distinct domains/groups are a proxy, "
                "not proof of independent reporting",
            },
        )
        contradiction = Signal(
            name="contradiction",
            # `combine()` pools every signal as "1.0 pushes toward SUPPORTED",
            # so the raw contradiction ratio has to be re-oriented around the
            # neutral 0.5:
            #   no contradiction     -> 0.5  (logit 0: no effect — absence of
            #                                evidence is not evidence of absence)
            #   everyone contradicts -> 0.0  (a strong downward pull)
            # Emitting the raw ratio meant a claim with NO contradiction scored
            # 0.0 and paid logit(0.01) * 0.20 ≈ -0.92 — the single largest term
            # in the pool, and the reason ~62% of the corpus sat at UNSUPPORTED.
            value=round(0.5 - 0.5 * contra_ratio, 3),
            weight=W_CONTRA,
            detail={
                "outlets_contradicting": sorted(contradicting_domains),
                "contradicting_syndication_groups": sorted(contradicting_groups),
                "contradiction_ratio": round(contra_ratio, 3),
            },
        )
        return corroboration, contradiction

    # -------------------------------------------------- track record

    @staticmethod
    def track_record_signal(
        outlet_stats: dict[str, Any] | None,
    ) -> Signal:
        """Optional smoothed independent counts; self-derived priors are rejected."""
        provenance = (outlet_stats or {}).get("provenance")
        if (
            not outlet_stats
            or outlet_stats.get("independently_validated") is not True
            or not isinstance(provenance, str)
            or not provenance.strip()
        ):
            return Signal(
                "source_track_record",
                0.5,
                W_TRACK,
                {
                    "prior_source": "neutral_no_independent_data",
                    "disclosure": "Neutral prior; the engine's own labels are "
                    "not outlet reliability evidence.",
                },
            )

        sup = float(outlet_stats.get("supported", 0))
        dis = float(outlet_stats.get("disputed", 0))
        if not all(math.isfinite(v) and v >= 0 for v in (sup, dis)):
            raise ValueError("Independent outlet counts must be finite and nonnegative")
        n = sup + dis
        smoothed = (sup + LAPLACE_ALPHA * 0.5) / (n + LAPLACE_ALPHA)
        return Signal(
            name="source_track_record",
            value=round(smoothed, 3),
            weight=W_TRACK,
            detail={
                "prior_source": "supplied_independent_data",
                "provenance": provenance,
                "disclosure": "Caller-supplied independent counts; provenance "
                "has not been audited by GMEE.",
                "outlet_supported_history": sup,
                "outlet_disputed_history": dis,
                "smoothing_alpha": LAPLACE_ALPHA,
                "sample_size": n,
            },
        )

    # -------------------------------------------------- combine

    @staticmethod
    def combine(signals: list[Signal]) -> VerdictResult:
        """Hand-weighted pooling, followed by evidence-gated legacy bands."""
        num = 0.0
        den = 0.0
        for s in signals:
            v = min(max(s.value, 0.01), 0.99)
            num += s.weight * math.log(v / (1 - v))
            den += s.weight
        p = 1 / (1 + math.exp(-num / den)) if den else 0.5
        p = round(min(max(p, 0.02), 0.98), 4)

        contra = next((s for s in signals if s.name == "contradiction"), None)
        corr = next((s for s in signals if s.name == "corroboration"), None)
        # The contradiction signal is neutral-centred (0.5 = no contradiction),
        # so DISPUTED is decided by the raw ratio carried in its detail.
        has_contra = bool(
            contra and float((contra.detail or {}).get("contradiction_ratio", 0.0)) > 0
        )
        has_corr = (
            corr is not None
            and len((corr.detail or {}).get("independent_outlets_supporting", [])) > 0
        )
        band = band_for(p, has_contra, has_corr)

        def contribution(signal: Signal) -> float:
            value = min(max(signal.value, 0.01), 0.99)
            return abs(signal.weight * math.log(value / (1 - value)))

        top = sorted(signals, key=contribution, reverse=True)[:2]
        parts = [
            f"{s.name} {'raises' if s.value >= 0.5 else 'lowers'} the score "
            f"({s.value:.2f})"
            for s in top
        ]
        rationale = f"Uncalibrated heuristic score={p:.2f} [{band}]."
        if parts:
            rationale += " Strongest factors: " + "; ".join(parts) + "."
        if not has_corr:
            rationale += " No cross-domain entailment established; this is not falsity."

        evidence: dict[str, Any] = {
            s.name: {"value": s.value, "weight": s.weight, **s.detail} for s in signals
        }
        evidence.update(
            {
                "score_kind": SCORE_KIND,
                "method_version": METHOD_VERSION,
                "nli": nli_configuration(),
                "warnings": [
                    ("The legacy probability field is an uncalibrated heuristic, "
                    "not a probability of truth."),
                    "Corpus-local textual stance is not independent fact verification.",
                    "Language and entity signals cannot establish factual support.",
                ],
            }
        )
        return VerdictResult(
            probability=p,
            band=band,
            rationale=rationale,
            evidence=evidence,
        )


# ------------------------------------------------------------- NLI stance


class NLIUnavailableError(RuntimeError):
    """The configured local classifier cannot run; do not manufacture a stance."""


@dataclass(frozen=True)
class StanceResult:
    stance: Stance
    abstention_reason: str | None = None


_nli_tok: Any = None
_nli_mdl: Any = None
_nli_model_key: tuple[str, str | None] | None = None
_nli_load_lock = Lock()
_nli_inference_lock = Lock()


def _load_nli(config: NLIConfiguration | None = None) -> tuple[Any, Any]:
    """Lazy, cache-local loading: serving a check never downloads model weights."""
    global _nli_tok, _nli_mdl, _nli_model_key
    cfg = config or nli_configuration()
    model_key = (cfg["model"], cfg["revision"])
    with _nli_load_lock:
        if _nli_mdl is None or _nli_model_key != model_key:
            try:
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                kwargs: dict[str, Any] = {"local_files_only": True}
                if cfg["revision"]:
                    kwargs["revision"] = cfg["revision"]
                logger.info("Loading cached three-class NLI model %s", cfg["model"])
                tok = AutoTokenizer.from_pretrained(cfg["model"], **kwargs)
                mdl = AutoModelForSequenceClassification.from_pretrained(
                    cfg["model"], **kwargs
                )
                mdl.eval()
            except Exception as exc:
                raise NLIUnavailableError(
                    f"NLI model '{cfg['model']}' is unavailable. Install the local "
                    "transformers/PyTorch/tokenizer dependencies and pre-cache the "
                    "configured model and NLI_MODEL_REVISION, or set NLI_MODEL to "
                    "a compatible local three-class NLI model directory. "
                    "Request-time model downloads are disabled."
                ) from exc
            _nli_tok, _nli_mdl, _nli_model_key = tok, mdl, model_key
    return _nli_tok, _nli_mdl


def _label_mapping(config: Any) -> dict[int, Stance] | None:
    """Require an explicit complete semantic mapping; LABEL_0 is not a stance."""
    mappings = [getattr(config, "id2label", {})]
    label2id = getattr(config, "label2id", {})
    if isinstance(label2id, dict):
        mappings.append({index: label for label, index in label2id.items()})
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        parsed: dict[int, Stance] = {}
        for index, label in mapping.items():
            normalized = str(label).strip().lower()
            if normalized not in ("entailment", "contradiction", "neutral"):
                break
            try:
                idx = int(index)
            except (ValueError, TypeError):
                break
            if normalized == "entailment":
                parsed[idx] = "entailment"
            elif normalized == "contradiction":
                parsed[idx] = "contradiction"
            else:
                parsed[idx] = "neutral"
        else:
            if set(parsed) == {0, 1, 2} and set(parsed.values()) == {
                "entailment",
                "contradiction",
                "neutral",
            }:
                return parsed
    return None


def _run_nli(
    claim_text: str, other_text: str, config: NLIConfiguration
) -> StanceResult:
    if not claim_text.strip() or not other_text.strip():
        return StanceResult("neutral", "empty_input")
    try:
        tok, mdl = _load_nli(config)
        labels = _label_mapping(mdl.config)
        if labels is None or getattr(mdl.config, "num_labels", 3) != 3:
            return StanceResult("neutral", "unknown_label_mapping")

        import torch

        # The premise is the evidence, the hypothesis is the checked claim.
        # Never silently drop a late negation or qualifier to fit the context.
        inputs = tok(other_text, claim_text, return_tensors="pt", truncation=False)
        context_limit = 512
        for limit in (
            getattr(tok, "model_max_length", 512),
            getattr(mdl.config, "max_position_embeddings", 512),
        ):
            if isinstance(limit, int) and 0 < limit < context_limit:
                context_limit = limit
        if inputs["input_ids"].shape[-1] > context_limit:
            return StanceResult("neutral", "input_exceeds_model_context")
        with _nli_inference_lock, torch.inference_mode():
            logits = mdl(**inputs).logits
            if tuple(logits.shape) != (1, 3):
                return StanceResult("neutral", "invalid_model_scores")
            scores = torch.softmax(logits, dim=-1)[0]
            if not bool(torch.isfinite(scores).all().item()):
                return StanceResult("neutral", "invalid_model_scores")
            confidence, index = scores.max(dim=0)
        if float(confidence.item()) < config["confidence_min"]:
            return StanceResult("neutral", "low_confidence")
        return StanceResult(labels[int(index.item())])
    except NLIUnavailableError:
        raise
    except Exception as exc:
        raise NLIUnavailableError(
            "Local NLI inference failed. Check NLI_MODEL/NLI_MODEL_REVISION, "
            "the three-class model files and PyTorch/tokenizer dependencies."
        ) from exc


STANCE_CACHE: OrderedDict[str, tuple[float, StanceResult]] = OrderedDict()
STANCE_CACHE_MAX = 20_000
STANCE_CACHE_TTL_SECONDS = 900.0


async def classify_stance(claim_text: str, other_text: str) -> StanceResult:
    """Canonical three-way stance with a bounded, full-text, configuration-keyed memo."""
    try:
        config = nli_configuration()
    except (TypeError, ValueError) as exc:
        raise NLIUnavailableError(f"Invalid local NLI configuration: {exc}") from exc
    key = hashlib.sha256(
        json.dumps(
            [METHOD_VERSION, config, claim_text, other_text],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cached = STANCE_CACHE.get(key)
    if cached:
        if monotonic() - cached[0] < STANCE_CACHE_TTL_SECONDS:
            STANCE_CACHE.move_to_end(key)
            return cached[1]
        del STANCE_CACHE[key]
    result = await asyncio.to_thread(_run_nli, claim_text, other_text, config)
    if STANCE_CACHE_MAX > 0:
        while len(STANCE_CACHE) >= STANCE_CACHE_MAX:
            STANCE_CACHE.popitem(last=False)
        STANCE_CACHE[key] = (monotonic(), result)
    return result


async def nli_stance(claim_text: str, other_text: str) -> LegacyStance:
    """Compatibility: yes=entailment, no=contradiction, neutral=abstention/neutral.

    A failure to entail is NOT a contradiction. No generative yes/no prompt is
    used, and unavailable models raise an actionable error rather than guess.
    """
    result = await classify_stance(claim_text, other_text)
    if result.stance == "entailment":
        return "yes"
    if result.stance == "contradiction":
        return "no"
    return "neutral"
