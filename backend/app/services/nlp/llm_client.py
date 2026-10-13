import asyncio
import json
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class ExtractedClaim(BaseModel):
    claim_text: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("claim_text")
    @classmethod
    def nonempty_claim(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Claim text must not be blank")
        return value


class LLMClient(ABC):
    @abstractmethod
    async def extract_claims(self, article_text: str) -> list[ExtractedClaim]:
        """Returns a list of extracted factual claims. Must return an empty
        list for articles with no extractable factual claims (e.g. opinion
        pieces) — this is a valid, successful result, not a failure."""

    @abstractmethod
    async def generate_mutation_summary(self, claims: list[str]) -> str:
        """Generates a natural language summary explaining how a claim mutated
        over time given a chronologically ordered list of claim texts."""

    @abstractmethod
    async def check_external_claim(self, claim: str, context: str) -> dict[str, str]:
        """Evaluates a claim against provided external context."""


class AnthropicLLMClient(LLMClient):
    def __init__(self) -> None:
        self.settings = get_settings()
        # Initialize only if API key is provided
        if self.settings.ANTHROPIC_API_KEY:
            self.client: AsyncAnthropic | None = AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)
        else:
            self.client = None

        self.model = getattr(self.settings, 'LLM_MODEL', 'claude-3-haiku-20240307')

    async def extract_claims(self, article_text: str) -> list[ExtractedClaim]:
        if not self.client:
            raise ValueError("ANTHROPIC_API_KEY is not configured")

        prompt = f"""
You are an expert fact-checker and information extraction system.
Extract all standalone factual claims from the following text.
If there are no factual claims (e.g. it is pure opinion, or too short), return an empty JSON array.

Guidelines for claims:
- Each claim should be a standalone sentence that can be verified independently.
- Resolve pronouns so the claim makes sense out of context.
- Provide a confidence score between 0.0 and 1.0 representing how clearly the text asserts this claim.
- IMPORTANT: If the text is NOT in English, you MUST translate the extracted claims into English. All returned claims MUST be in English.

Return ONLY a JSON array of objects, with no markdown formatting, no code blocks, and no preamble.
The JSON objects must have exactly these two keys:
1. "claim_text" (string)
2. "confidence" (number)

Example output format:
[
    {{"claim_text": "The Eiffel Tower is located in Paris, France.", "confidence": 0.99}},
    {{"claim_text": "The company's revenue increased by 15% in Q3 2023.", "confidence": 0.95}}
]

Text to analyze:
{article_text}
"""

        try:
            params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 1024,
                "temperature": 0.0,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }
            response = await self.client.messages.create(**params)

            raw_text = response.content[0].text.strip()

            # Try to parse the JSON
            try:
                data = json.loads(raw_text)
                if not isinstance(data, list):
                    raise TypeError("Expected a JSON array")

                claims = []
                for item in data:
                    claims.append(ExtractedClaim.model_validate(item))
                return claims

            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as parse_err:
                logger.error(f"Failed to parse LLM response. Error: {parse_err}. Raw response (truncated): {raw_text[:200]}")
                raise ValueError(f"Invalid LLM response format: {parse_err}")

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def generate_mutation_summary(self, claims: list[str]) -> str:
        if not self.client:
            raise ValueError("ANTHROPIC_API_KEY is not configured")
        
        claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(claims)])
        prompt = f"""
You are an expert misinformation analyst. The following is a chronological sequence of claims from different news sources that belong to the same rumor/story.
Please write a concise 2-sentence summary explaining exactly how the claim mutated or evolved over time (e.g. "Sources initially reported X, but later sources exaggerated it to Y"). Do not evaluate truthfulness, just summarize the mutation.

Chronological Claims:
{claims_text}
"""
        try:
            params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 150,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            }
            response = await self.client.messages.create(**params)
            return str(response.content[0].text).strip()
        except Exception as e:
            logger.error(f"Anthropic API error during mutation summary: {e}")
            return "Unable to generate summary due to API error."

    async def check_external_claim(self, claim: str, context: str) -> dict[str, str]:
        if not self.client:
            raise ValueError("ANTHROPIC_API_KEY is not configured")
            
        prompt = f"""
You are an expert fact-checker. Determine if the following claim is Supported, Disputed, or Unverifiable based solely on the provided external context.

Claim: "{claim}"

Context:
{context}

Respond EXACTLY in this JSON format:
{{"verdict": "SUPPORTED|DISPUTED|UNVERIFIABLE", "explanation": "1-2 sentences explaining why, referencing the source"}}
"""
        try:
            params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 150,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            }
            response = await self.client.messages.create(**params)
            raw = response.content[0].text.strip()
            # fallback parsing if markdown
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            data = json.loads(raw)
            return {"verdict": data.get("verdict", "UNVERIFIABLE"), "explanation": data.get("explanation", "")}
        except Exception as e:
            logger.error(f"Anthropic API error during external check: {e}")
            return {"verdict": "UNVERIFIABLE", "explanation": "LLM validation failed."}


class HuggingFaceLLMClient(LLMClient):
    """Local zero-cost claim extractor backed by an open HF model.

    Runs FLAN-T5-base instruction-tuned model on CPU via transformers'
    text2text pipeline. Same interface as the Anthropic client, so the NLP
    orchestrator can use either transparently. Preference order in
    get_llm_client(): explicit ANTHROPIC key > local HF model.
    """

    MODEL_ID = os.environ.get("HF_LLM_MODEL", "google/flan-t5-base")
    BATCH_SIZE = 8
    _loaded_pipes: ClassVar[dict[str, tuple[Any, Any, str]]] = {}
    _load_lock = threading.Lock()
    _inference_lock = threading.Lock()

    def __init__(self) -> None:
        self._pipe: tuple[Any, Any, str] | None = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        # Orchestrators are short-lived; share the expensive model between them.
        # The lock also prevents concurrent first requests loading duplicate copies.
        with self._load_lock:
            if self.MODEL_ID in self._loaded_pipes:
                self._pipe = self._loaded_pipes[self.MODEL_ID]
                return
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading HF claim-extraction model %s …", self.MODEL_ID)
            tok = AutoTokenizer.from_pretrained(self.MODEL_ID)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_ID)
            mdl.to(device).eval()
            self._pipe = (tok, mdl, device)
            self._loaded_pipes[self.MODEL_ID] = self._pipe
            logger.info("HF claim-extraction model loaded.")

    def _split_sentences(self, text: str) -> list[str]:
        # Keep short factual sentences and long ones; neither length nor article
        # position determines whether a sentence deserves analysis. A lightweight
        # English tokenizer handles abbreviations ("Dr.", "U.S.") and decimals.
        import spacy

        segmenter = spacy.blank("en")
        segmenter.add_pipe("sentencizer")
        segmenter.max_length = max(segmenter.max_length, len(text) + 1)
        return [span.text.strip() for span in segmenter(text).sents
                if len(span.text.split()) >= 3 and any(ch.isalpha() for ch in span.text)]

    def _score_claim(self, sentence: str) -> float:
        """Heuristic confidence from structural verifiability signals."""
        base = 0.5
        if any(ch.isdigit() for ch in sentence):
            base += 0.15
        if any(
            k in sentence.lower()
            for k in ("said", "according to", "announced", "reported", "confirmed", "stated")
        ):
            base += 0.12
        words = sentence.split()
        if len(words) >= 10:
            base += 0.08
        low = f" {sentence.lower()} "
        if any(
            k in low
            for k in (" i ", " my ", " we ", " our ", "think", "feel", "love", "hate", "ridiculous")
        ):
            base -= 0.3
        return round(min(0.97, max(0.05, base)), 2)

    async def extract_claims(self, article_text: str) -> list[ExtractedClaim]:
        def run_pipeline() -> list[ExtractedClaim]:
            import torch

            sentences = self._split_sentences(article_text)
            if not sentences:
                return []
            self._load()
            pipe = self._pipe
            assert pipe is not None
            tok, mdl, device = pipe
            # Classify bounded overlapping windows for every sentence, including
            # the end of long articles. Return original source sentences, never
            # generated facts. This is check-worthiness, not truth verification.
            windows: list[tuple[int, str]] = []
            for index, sentence in enumerate(sentences):
                words = sentence.split()
                for start in range(0, len(words), 120):
                    window = " ".join(words[start:start + 144])
                    windows.append((index, (
                        f"Classify the sentence as factual or opinion: {window}"
                    )))
                    if start + 144 >= len(words):
                        break
            accepted: set[int] = set()
            for start in range(0, len(windows), self.BATCH_SIZE):
                batch = windows[start:start + self.BATCH_SIZE]
                prompts = [prompt for _, prompt in batch]
                # Generation shares model state; serialize it across HTTP/jobs.
                with self._inference_lock, torch.inference_mode():
                    inputs = tok(prompts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    out_ids = mdl.generate(**inputs, max_new_tokens=6, do_sample=False)
                    decoded = tok.batch_decode(out_ids, skip_special_tokens=True)
                for (index, _), answer in zip(batch, decoded, strict=True):
                    if re.fullmatch(r"(?:factual|yes)[.!]?", str(answer).strip(), re.IGNORECASE):
                        accepted.add(index)
            claims: list[ExtractedClaim] = []
            seen: set[str] = set()
            for index, sentence in enumerate(sentences):
                confidence = self._score_claim(sentence)
                # The structural score is disclosed metadata, not a second
                # classifier: Roman numerals/names such as Artemis I must not
                # be discarded by the first-person heuristic.
                if index in accepted and not sentence.endswith("?") and sentence not in seen:
                    claims.append(ExtractedClaim(claim_text=sentence, confidence=confidence))
                    seen.add(sentence)
            return claims

        try:
            return await asyncio.to_thread(run_pipeline)
        except Exception as e:
            logger.error(f"HF pipeline error: {e}")
            raise

    async def generate_mutation_summary(self, claims: list[str]) -> str:
        from app.services.evolution.text_changes import analyze_text_change

        if not claims:
            return "No claims are available to compare."
        if len(claims) == 1:
            return "One claim is available; at least two are needed to describe changes."
        analysis = analyze_text_change(claims[0], claims[-1])
        details = "; ".join(
            f"{change['older_span']['text'] or '(absent)'} → "
            f"{change['newer_span']['text'] or '(absent)'}"
            for change in analysis["changed_spans"][:4]
        )
        change_text = f"Text changes: {details}. " if details else ""
        return (
            f"Compared the first and last of {len(claims)} ordered claims. "
            f"{change_text}{analysis['relation_explanation']} "
            "This text comparison does not establish truth or transmission."
        )

    async def check_external_claim(self, claim: str, context: str) -> dict[str, str]:
        from app.services.verdict.engine import NLIUnavailableError, classify_stance

        if not claim.strip() or not context.strip():
            return {"verdict": "UNVERIFIABLE", "explanation": "A claim and evidence passage are required."}
        try:
            result = await classify_stance(claim, context)
        except NLIUnavailableError:
            return {"verdict": "UNVERIFIABLE", "explanation": "The local stance model is unavailable; prepare its cached weights."}
        verdict = {"entailment": "SUPPORTED", "contradiction": "DISPUTED", "neutral": "UNVERIFIABLE"}[result.stance]
        return {"verdict": verdict, "explanation": (
            f"The supplied passage has {result.stance} stance toward this claim. "
            "Textual stance does not establish independent factual truth."
            + (f" Assessment abstained: {result.abstention_reason}." if result.abstention_reason else "")
        )}


def get_llm_client() -> LLMClient:
    """Factory used by the NLP orchestrator.

    Priority:
      1. Anthropic when ANTHROPIC_API_KEY is set (best quality)
      2. Local HuggingFace model otherwise (zero cost, offline)
         - set LLM_BACKEND=none to force the legacy skip behaviour
    """
    settings = get_settings()
    backend = getattr(settings, "LLM_BACKEND", "") or (
        "anthropic" if settings.ANTHROPIC_API_KEY else "huggingface"
    )
    if backend == "anthropic" and settings.ANTHROPIC_API_KEY:
        return AnthropicLLMClient()
    if backend == "none":
        return _NoOpLLMClient()
    return HuggingFaceLLMClient()


class _NoOpLLMClient(LLMClient):
    """Legacy behaviour: no extraction possible → mark articles skipped."""

    async def extract_claims(self, article_text: str) -> list[ExtractedClaim]:
        return []

    async def generate_mutation_summary(self, claims: list[str]) -> str:
        return "Mutation summary unavailable (LLM backend is none)."

    async def check_external_claim(self, claim: str, context: str) -> dict[str, str]:
        return {"verdict": "UNVERIFIABLE", "explanation": "LLM backend is none."}
