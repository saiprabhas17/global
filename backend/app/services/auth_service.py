import uuid
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import redis_helpers, security
from app.core.config import get_settings
from app.models.user import RoleEnum, User
from app.repositories import session_repository, user_repository
from app.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    TokenResponse,
    UserCreate,
)

settings = get_settings()

credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

generic_error = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Incorrect email or password",
    headers={"WWW-Authenticate": "Bearer"},
)

async def _issue_tokens_and_session(db: AsyncSession, user: User, device_info: str | None) -> TokenResponse:
    # 1. Access Token
    jti = str(uuid.uuid4())
    access_token = security.create_access_token(user_id=str(user.id), role=user.role.value, jti=jti)
    
    # 2. Refresh Token
    refresh_token = security.generate_refresh_token()
    refresh_token_hash = security.hash_refresh_token(refresh_token)
    
    expires_at = datetime.now(UTC) + timedelta(days=settings.REFRESH_TOKEN_TTL_DAYS)
    
    # 3. Create Session
    await session_repository.create(db, user.id, refresh_token_hash, device_info, expires_at)
    await db.commit()
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )

async def register_user(db: AsyncSession, user_in: UserCreate, device_info: str | None) -> TokenResponse:
    user = await user_repository.get_by_email(db, user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )
        
    hashed_password = security.hash_password(user_in.password)
    new_user = await user_repository.create(db, user_in.email, hashed_password, user_in.full_name, RoleEnum.user)
    
    from sqlalchemy import exc
    try:
        await db.commit()
        await db.refresh(new_user)
    except exc.IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )
        
    return await _issue_tokens_and_session(db, new_user, device_info)

async def authenticate_user(db: AsyncSession, redis: Redis, login_in: LoginRequest, ip: str, device_info: str | None) -> TokenResponse:
    # Rate Limiting
    ttl = await redis_helpers.check_login_rate_limit(redis, login_in.email, ip)
    if ttl > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Try again later.",
            headers={"Retry-After": str(ttl)}
        )
        
    user = await user_repository.get_by_email(db, login_in.email)
    if not user or not user.is_active or not security.verify_password(user.hashed_password, login_in.password):
        raise generic_error
        
    return await _issue_tokens_and_session(db, user, device_info)

async def rotate_refresh_token(db: AsyncSession, refresh_in: RefreshRequest, device_info: str | None) -> TokenResponse:
    token_hash = security.hash_refresh_token(refresh_in.refresh_token)
    db_session = await session_repository.get_by_refresh_token_hash(db, token_hash, for_update=True)
    
    if not db_session:
        raise credentials_exception
        
    # Reuse Detection
    if db_session.revoked:
        await session_repository.revoke_all_for_user(db, db_session.user_id, datetime.now(UTC))
        await db.commit()
        raise credentials_exception
        
    # Check expiry
    expires_at = db_session.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
        
    now_utc = datetime.now(UTC)
    if expires_at < now_utc:
        await session_repository.revoke(db, db_session, now_utc)
        await db.commit()
        raise credentials_exception
        
    # Get user
    user = await user_repository.get_by_id(db, db_session.user_id)
    if not user or not user.is_active:
        raise credentials_exception
        
    # Rotate token
    await session_repository.revoke(db, db_session, now_utc)
    
    return await _issue_tokens_and_session(db, user, device_info)

async def logout_user(db: AsyncSession, redis: Redis, logout_in: LogoutRequest, current_user: User, auth_header: str | None) -> None:
    token_hash = security.hash_refresh_token(logout_in.refresh_token)
    db_session = await session_repository.get_by_refresh_token_hash(db, token_hash, for_update=True)
    
    now_utc = datetime.now(UTC)
    if db_session and not db_session.revoked and db_session.user_id == current_user.id:
        await session_repository.revoke(db, db_session, now_utc)
        await db.commit()
        
    # Blocklist access token
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        payload = security.decode_access_token(token)
        if payload and "jti" in payload and "exp" in payload:
            jti = payload["jti"]
            expires_at = datetime.fromtimestamp(payload["exp"], tz=UTC)
            await redis_helpers.blocklist_access_token(redis, jti, expires_at)
