import logging
import threading

import spacy
from pydantic import BaseModel
from spacy.language import Language

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    entity_text: str
    entity_type: str
    start_char: int
    end_char: int


class EntityExtractor:
    _nlp: Language | None = None
    _lock = threading.Lock()

    @classmethod
    def load_model(cls) -> None:
        """Load spaCy once (idempotent, thread-safe); not for request paths."""
        if cls._nlp is not None:
            return
        with cls._lock:
            if cls._nlp is None:
                logger.info("Loading spaCy model en_core_web_sm...")
                cls._nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded.")

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._nlp is not None

    @classmethod
    def extract_entities(cls, text: str, *, allow_load: bool = False) -> list[ExtractedEntity]:
        if cls._nlp is None:
            if not allow_load:
                raise RuntimeError("spaCy model not loaded. Call load_model() first.")
            cls.load_model()
        assert cls._nlp is not None
        
        doc = cls._nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append(
                ExtractedEntity(
                    entity_text=ent.text,
                    entity_type=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                )
            )
        return entities

    @classmethod
    def extract_keywords(cls, text: str) -> list[str]:
        """Extract noun chunks as lightweight keywords."""
        if cls._nlp is None:
            cls.load_model()
            
        assert cls._nlp is not None
        doc = cls._nlp(text)
        # Using a set to deduplicate noun chunks (case-insensitive)
        keywords = {chunk.text.lower() for chunk in doc.noun_chunks}
        return list(keywords)
