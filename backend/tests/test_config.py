import os

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_cors_origins_parsing() -> None:
    os.environ["POSTGRES_URL"] = "postgresql+asyncpg://postgres:postgres@localhost/test"
    os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
    os.environ["NEO4J_USER"] = "test"
    os.environ["NEO4J_PASSWORD"] = "test"
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    os.environ["CORS_ORIGINS"] = "http://localhost:3000,http://localhost:5173"
    
    settings = Settings()  # type: ignore
    
    assert settings.CORS_ORIGINS == ["http://localhost:3000", "http://localhost:5173"]

def test_settings_cors_origins_json_parsing() -> None:
    os.environ["POSTGRES_URL"] = "postgresql+asyncpg://postgres:postgres@localhost/test"
    os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
    os.environ["NEO4J_USER"] = "test"
    os.environ["NEO4J_PASSWORD"] = "test"
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    os.environ["CORS_ORIGINS"] = '["http://localhost:3000", "http://localhost:5173"]'
    
    settings = Settings()  # type: ignore
    
    assert settings.CORS_ORIGINS == ["http://localhost:3000", "http://localhost:5173"]


_BASE_ENV = {
    "POSTGRES_URL": "postgresql+asyncpg://postgres:postgres@localhost/test",
    "CORS_ORIGINS": "http://localhost:3000",
    "JWT_SECRET": "super_secret_test_jwt_key_that_is_at_least_32_chars",
}


def test_newsapi_key_accepts_both_spellings(monkeypatch) -> None:
    """Regression: the code reads ``NEWSAPI_KEY`` while ``.env.example`` and the
    README documented ``NEWS_API_KEY``. Because Settings uses ``extra="ignore"``
    the documented spelling was dropped and the NewsAPI collector could never be
    enabled."""
    for key, value in _BASE_ENV.items():
        monkeypatch.setenv(key, value)

    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    monkeypatch.setenv("NEWS_API_KEY", "documented-spelling")
    assert Settings().NEWSAPI_KEY == "documented-spelling"  # type: ignore[call-arg]

    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    monkeypatch.setenv("NEWSAPI_KEY", "canonical-spelling")
    assert Settings().NEWSAPI_KEY == "canonical-spelling"  # type: ignore[call-arg]


def test_llm_backend_is_a_declared_setting(monkeypatch) -> None:
    """Regression: ``get_llm_client()`` read ``LLM_BACKEND`` through
    ``getattr(settings, ...)`` but Settings never declared the field, so the
    documented ``LLM_BACKEND=none`` switch was a silent no-op."""
    for key, value in _BASE_ENV.items():
        monkeypatch.setenv(key, value)

    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert Settings().LLM_BACKEND == ""  # type: ignore[call-arg]

    monkeypatch.setenv("LLM_BACKEND", "none")
    assert Settings().LLM_BACKEND == "none"  # type: ignore[call-arg]


def test_verdict_calibration_knobs_are_declared(monkeypatch) -> None:
    for key, value in _BASE_ENV.items():
        monkeypatch.setenv(key, value)

    settings = Settings()  # type: ignore
    assert 0.0 < settings.VERDICT_STRONG_SIMILARITY <= 1.0
    assert settings.VERDICT_NEAR_MIN < settings.VERDICT_NEAR_MAX


def test_nli_configuration_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _BASE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("NLI_MODEL", "example/stance-model")
    monkeypatch.setenv("NLI_MODEL_REVISION", "reviewed-revision")
    monkeypatch.setenv("NLI_CONFIDENCE_MIN", "0.85")
    settings = Settings()  # type: ignore[call-arg]
    assert settings.NLI_MODEL == "example/stance-model"
    assert settings.NLI_MODEL_REVISION == "reviewed-revision"
    assert settings.NLI_CONFIDENCE_MIN == 0.85
    for invalid in ("nan", "inf", "-0.1", "1.1"):
        monkeypatch.setenv("NLI_CONFIDENCE_MIN", invalid)
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]
