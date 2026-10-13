"""A single-use refresh token cannot be rotated twice under PostgreSQL concurrency."""
import asyncio
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, select

from app.db.postgres import async_session_maker, engine
from app.models.session import Session
from app.models.user import User
from app.schemas.auth import RefreshRequest, TokenResponse, UserCreate
from app.services.auth_service import register_user, rotate_refresh_token

pytestmark = pytest.mark.integration


async def test_concurrent_refresh_is_single_use_and_replay_revokes_family() -> None:
    await engine.dispose()
    user_id = None
    email = f"refresh-fixture-{uuid.uuid4().hex}@example.com"
    try:
        async with async_session_maker() as db:
            issued = await register_user(db, UserCreate(email=email, password="Fixture-only-strong-password!"), None)
            user_id = await db.scalar(select(User.id).where(User.email == email))
            assert user_id is not None
        ready = asyncio.Event()

        async def rotate() -> TokenResponse | HTTPException:
            async with async_session_maker() as db:
                await ready.wait()
                try:
                    return await rotate_refresh_token(db, RefreshRequest(refresh_token=issued.refresh_token), None)
                except HTTPException as exc:
                    return exc

        first, second = asyncio.create_task(rotate()), asyncio.create_task(rotate())
        ready.set()
        results = await asyncio.gather(first, second)
        assert sum(isinstance(result, TokenResponse) for result in results) == 1
        failures = [result for result in results if isinstance(result, HTTPException)]
        assert len(failures) == 1 and failures[0].status_code == 401
        async with async_session_maker() as db:
            sessions = (await db.scalars(select(Session).where(Session.user_id == user_id))).all()
            assert len(sessions) == 2
            assert all(session.revoked for session in sessions)
    finally:
        if user_id is not None:
            async with async_session_maker() as db:
                await db.execute(delete(Session).where(Session.user_id == user_id))
                await db.execute(delete(User).where(User.id == user_id))
                await db.commit()
        await engine.dispose()
