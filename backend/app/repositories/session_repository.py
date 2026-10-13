import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.session import Session


async def get_by_refresh_token_hash(
    db: AsyncSession, token_hash: str, *, for_update: bool = False
) -> Session | None:
    statement = select(Session).where(Session.refresh_token_hash == token_hash)
    if for_update:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    result = await db.execute(statement)
    return result.scalars().first()

async def create(db: AsyncSession, user_id: uuid.UUID, refresh_token_hash: str, device_info: str | None, expires_at: datetime) -> Session:
    new_session = Session(
        user_id=user_id,
        refresh_token_hash=refresh_token_hash,
        device_info=device_info,
        expires_at=expires_at
    )
    db.add(new_session)
    return new_session

async def revoke(db: AsyncSession, session: Session, revoked_at: datetime) -> None:
    session.revoked = True
    session.revoked_at = revoked_at

async def revoke_all_for_user(db: AsyncSession, user_id: uuid.UUID, revoked_at: datetime) -> None:
    await db.execute(
        update(Session)
        .where(Session.user_id == user_id, Session.revoked == False)
        .values(revoked=True, revoked_at=revoked_at)
    )
