from functools import lru_cache
from typing import Any, Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    POSTGRES_URL: str
    
    @field_validator("POSTGRES_URL", mode="before")
    @classmethod
    def fix_postgres_scheme(cls, v: Any) -> Any:
        """Render provides postgres:// but asyncpg needs postgresql+asyncpg://."""
        if isinstance(v, str):
            if v.startswith("postgres://"):
                v = v.replace("postgres://", "postgresql+asyncpg://", 1)
            elif v.startswith("postgresql://"):
                v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    NEO4J_URI: str = ""
    NEO4J_USER: str = ""
    NEO4J_PASSWORD: str = ""
    REDIS_URL: str = ""
    CORS_ORIGINS: str | list[str]
    ENVIRONMENT: Literal["development", "production", "testing"] = "development"
    
    JWT_SECRET: str
    ACCESS_TOKEN_TTL_MINUTES: int = 15
    REFRESH_TOKEN_TTL_DAYS: int = 7

    # Data Collection
    COLLECTION_INTERVAL_MINUTES: int = 30
    # Accepts both NEWSAPI_KEY (the canonical attribute name) and NEWS_API_KEY
    # (the name used by .env.example / README). A plain str field would silently
    # ignore NEWS_API_KEY because Settings uses extra="ignore".
    NEWSAPI_KEY: str = Field(
        default="",
        validation_alias=AliasChoices("NEWSAPI_KEY", "NEWS_API_KEY"),
    )
    REDDIT_CLIENT_ID: str = ""
    REDDIT_CLIENT_SECRET: str = ""
    REDDIT_USER_AGENT: str = "python:gmee:v0.1.0"
    
    # Preprocessing
    NEAR_DUP_JACCARD_THRESHOLD: float = 0.75
    NEAR_DUP_WINDOW_DAYS: int = 14
    
    # NLP
    ANTHROPIC_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""
    LLM_MODEL: str = "claude-haiku-4-5-20251001"
    NLP_MAX_ARTICLES_PER_CYCLE: int = 10
    # Claim-extraction backend. "" (default) auto-selects: Anthropic when an
    # API key is present, otherwise the local HuggingFace FLAN-T5 extractor.
    # Set to "anthropic", "huggingface" or "none" (legacy skip behaviour).
    LLM_BACKEND: str = ""

    # Evolution Engine
    MIN_CORPUS_SIZE_FOR_CLUSTERING: int = 30
    MIN_NEW_CLAIMS_TO_RECLUSTER: int = 10
    EVOLUTION_SIMILARITY_THRESHOLD: float = 0.85
    SIMILAR_TO_THRESHOLD: float = 0.75
    BERTOPIC_MIN_TOPIC_SIZE: int = 5

    # Verdict engine calibration
    # Cosine-similarity window in which two claims are considered "the same
    # assertion". The original hard-coded 0.75 floor proved too strict on real
    # news corpora (cross-outlet paraphrases of one event typically sit at
    # 0.55–0.75), collapsing ~95% of verdicts to UNSUPPORTED.
    VERDICT_NEAR_MIN: float = 0.60
    VERDICT_NEAR_MAX: float = 0.97

    # Cosine similarity at or above which a cross-outlet neighbour is counted
    # as corroboration WITHOUT waiting for the NLI model to say "yes". At this
    # distance the two sentences are near-identical paraphrases, and FLAN-T5
    # base was measured to answer "neutral" for most of them — which starved
    # the corroboration signal and collapsed the corpus to UNSUPPORTED.
    VERDICT_STRONG_SIMILARITY: float = 0.85

    # Embedding model. Default: all-mpnet-base-v2 (768-dim, English).
    # Cross-lingual lineage: set to paraphrase-multilingual-mpnet-base-v2
    # (also 768-dim, 50+ languages) — a drop-in swap for existing vector
    # columns. NOTE: changing the model after data exists requires
    # re-embedding the whole corpus (scripts/embed_articles.py) or the
    # vector spaces are incomparable.
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"
    NLI_MODEL_REVISION: str | None = None
    NLI_CONFIDENCE_MIN: float = Field(default=0.7, ge=0.0, le=1.0, allow_inf_nan=False)

    # Process model: run the APScheduler inside this process (dev / single
    # node). Set to false for API-only replicas and run scripts/run_nlp_cycle.py
    # as a separate process so heavy cycles never restart with the API.
    ENABLE_SCHEDULER: bool = False

    # Rate limiting. Disable in the test suite (tests share one IP and would
    # trip the login limiter against each other / against dev usage).
    RATE_LIMIT_ENABLED: bool = True
    # When Redis is unreachable, sensitive endpoints (login, register, admin
    # triggers) return 503 instead of silently dropping their brute-force guard.
    # Ordinary read endpoints still fail open to keep the dashboard available.
    RATE_LIMIT_FAIL_CLOSED_SENSITIVE: bool = True

    # Proxy trust. Empty (default) = the raw X-Forwarded-For header is IGNORED
    # and the socket peer (already rewritten by Uvicorn --forwarded-allow-ips)
    # is the client identity. Only list networks you operate, e.g. "10.0.0.0/8".
    TRUSTED_PROXY_CIDRS: str | list[str] = []

    # Scheduler leader election. False = if Redis is unavailable no replica
    # runs scheduled work (fail closed) instead of every replica running it.
    LEADER_LOCK_FAIL_OPEN: bool = False

    # Durable pipeline ownership. A worker owns claimed articles/jobs until the
    # lease expires; expired claims are returned to the queue at most
    # PIPELINE_MAX_ATTEMPTS times, then marked failed for operator review.
    PIPELINE_LEASE_SECONDS: int = 900
    PIPELINE_MAX_ATTEMPTS: int = 3
    PIPELINE_RECOVERY_BATCH: int = 500

    @field_validator("JWT_SECRET")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters long")
        return v

    @field_validator("CORS_ORIGINS", "TRUSTED_PROXY_CIDRS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    @field_validator("TRUSTED_PROXY_CIDRS")
    @classmethod
    def validate_trusted_proxies(cls, v: Any) -> list[str]:
        import ipaddress

        networks = [v] if isinstance(v, str) else list(v or [])
        for cidr in networks:
            network = ipaddress.ip_network(cidr, strict=False)
            if network.prefixlen == 0:
                raise ValueError("TRUSTED_PROXY_CIDRS must not trust every address")
        return networks

    @field_validator("PIPELINE_LEASE_SECONDS")
    @classmethod
    def validate_lease(cls, v: int) -> int:
        if not 30 <= v <= 86_400:
            raise ValueError("PIPELINE_LEASE_SECONDS must be between 30 and 86400")
        return v

    @field_validator("PIPELINE_MAX_ATTEMPTS")
    @classmethod
    def validate_attempts(cls, v: int) -> int:
        if not 1 <= v <= 20:
            raise ValueError("PIPELINE_MAX_ATTEMPTS must be between 1 and 20")
        return v

    # Loads from local .env or parent root .env if running outside container
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore
