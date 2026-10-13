"""Auth service flows: register → login → rotate → reuse-detection → logout."""
import pytest
from fastapi import HTTPException

from app.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    UserCreate,
)
from app.services.auth_service import (
    authenticate_user,
    logout_user,
    register_user,
    rotate_refresh_token,
)

PW = "Str0ngPass!x"


async def test_inactive_user_cannot_log_in(db_session, redis_client):
    from sqlalchemy import select

    from app.models.user import User

    await register_user(db_session, UserCreate(email="inactive@example.com", password=PW), None)
    user = await db_session.scalar(select(User).where(User.email == "inactive@example.com"))
    assert user is not None
    user.is_active = False
    await db_session.commit()
    with pytest.raises(HTTPException) as failure:
        await authenticate_user(db_session, redis_client,
            LoginRequest(email="inactive@example.com", password=PW), "127.0.0.1", None)
    assert failure.value.status_code == 401


async def test_register_issues_token_pair(db_session):
    res = await register_user(
        db_session,
        UserCreate(email="flow@example.com", password=PW, full_name="Flow"),
        device_info="pytest",
    )
    assert res.access_token and res.refresh_token


async def test_register_duplicate_email_conflicts(db_session):
    await register_user(db_session, UserCreate(email="dup@example.com", password=PW), None)
    with pytest.raises(HTTPException) as ei:
        await register_user(db_session, UserCreate(email="dup@example.com", password=PW), None)
    assert ei.value.status_code == 409


async def test_login_success_and_wrong_password_generic_error(db_session, redis_client):
    await register_user(db_session, UserCreate(email="login@example.com", password=PW), None)

    ok = await authenticate_user(
        db_session, redis_client,
        LoginRequest(email="login@example.com", password=PW),
        ip="127.0.0.1", device_info="pytest",
    )
    assert ok.access_token

    with pytest.raises(HTTPException) as ei:
        await authenticate_user(
            db_session, redis_client,
            LoginRequest(email="login@example.com", password="wrong-pass"),
            ip="127.0.0.1", device_info="pytest",
        )
    assert ei.value.status_code == 401
    assert ei.value.detail == "Incorrect email or password"


async def test_refresh_rotation_and_reuse_detection(db_session, redis_client):
    created = await register_user(
        db_session, UserCreate(email="rot@example.com", password=PW), None
    )

    rotated = await rotate_refresh_token(
        db_session, RefreshRequest(refresh_token=created.refresh_token), "pytest"
    )
    assert rotated.refresh_token != created.refresh_token

    # Reusing the now-revoked token must revoke ALL sessions for the user
    with pytest.raises(HTTPException) as ei:
        await rotate_refresh_token(
            db_session, RefreshRequest(refresh_token=created.refresh_token), "pytest"
        )
    assert ei.value.status_code == 401

    # …and the newer rotated token is dead too (full family revoked)
    with pytest.raises(HTTPException):
        await rotate_refresh_token(
            db_session, RefreshRequest(refresh_token=rotated.refresh_token), "pytest"
        )


async def test_logout_revokes_and_blocklists(db_session, redis_client):
    await register_user(
        db_session, UserCreate(email="out@example.com", password=PW), None
    )
    me = await authenticate_user(
        db_session, redis_client,
        LoginRequest(email="out@example.com", password=PW),
        ip="127.0.0.1", device_info="pytest",
    )

    # current_user comes from the access token's user; fetch via login flow:
    from app.models.user import User  # local import keeps top clean
    from app.repositories import user_repository

    user = await user_repository.get_by_email(db_session, "out@example.com")
    assert isinstance(user, User)

    await logout_user(
        db_session, redis_client,
        LogoutRequest(refresh_token=me.refresh_token),
        current_user=user,
        auth_header=f"Bearer {me.access_token}",
    )

    with pytest.raises(HTTPException):
        await rotate_refresh_token(
            db_session, RefreshRequest(refresh_token=me.refresh_token), "pytest"
        )
