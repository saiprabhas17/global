from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.POSTGRES_URL,
    # SQL parameters contain password hashes, token hashes and private text.
    # Development must follow the same credential-redaction policy as serving.
    echo=False,
    future=True
)

async_session_maker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)
