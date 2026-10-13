"""Redis-backed leader election for scheduled jobs.

When multiple backend replicas run, each has its own APScheduler. Without
coordination every replica fires the same collection/NLP cycle, double-
fetching sources and racing on the same rows.

This lock makes only ONE replica per interval execute a job:
  - SET key token NX PX ttl  ->  won by exactly one caller
  - losers skip silently and retry next tick
  - the owner renews the lease while work runs (``hold``), so a long cycle
    does not outlive its lock and let a second leader start
  - release and renewal are compare-and-set on the owner token (WATCH/MULTI),
    so an old leader can never delete or extend a newer leader's lock
  - stale locks expire, so a crashed leader cannot wedge the pipeline

Failure policy: if Redis is unreachable, acquisition FAILS CLOSED (no replica
runs) unless ``LEADER_LOCK_FAIL_OPEN`` is set. Redis election is a scheduling
optimisation, not exactly-once execution: durable row ownership in Postgres
(``app.services.pipeline``) is what prevents two workers processing one row.
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class LeaderLease:
    """State of a held lock; ``lost`` flips when renewal fails."""

    job: str
    token: str
    acquired: bool
    lost: asyncio.Event = field(default_factory=asyncio.Event)
    coordinated: bool = True  # False when running uncoordinated (fail-open)

    @property
    def is_lost(self) -> bool:
        return self.lost.is_set()


class LeaderLock:
    def __init__(self, redis_wrapper: Any, namespace: str = "gmee") -> None:
        # ``redis_wrapper`` is the app's redis_client wrapper (has get_client())
        self._wrapper = redis_wrapper
        self.namespace = namespace
        self.instance_id = uuid.uuid4().hex[:12]

    def _key(self, job: str) -> str:
        return f"{self.namespace}:lock:{job}"

    async def _client(self) -> Redis:
        client: Redis = await self._wrapper.get_client()
        return client

    def _token(self) -> str:
        # Unique per acquisition so an old holder's token never matches.
        return f"{self.instance_id}:{uuid.uuid4().hex[:12]}"

    async def _acquire_token(self, job: str, ttl_seconds: int) -> tuple[bool, str, bool]:
        """Return (acquired, token, coordinated)."""
        token = self._token()
        try:
            client = await self._client()
            acquired = await client.set(self._key(job), token, nx=True, px=int(ttl_seconds * 1000))
            return bool(acquired), token, True
        except Exception as exc:
            if get_settings().LEADER_LOCK_FAIL_OPEN:
                logger.warning(
                    "leader lock unavailable (%s) — proceeding UNCOORDINATED by configuration",
                    type(exc).__name__,
                )
                return True, token, False
            logger.error(
                "leader lock unavailable (%s) — skipping %s (fail closed)",
                type(exc).__name__,
                job,
            )
            return False, token, True

    async def try_acquire(self, job: str, ttl_seconds: int) -> bool:
        """Attempt to become the leader for ``job``. True when acquired.

        Prefer :meth:`hold`, which also renews and releases safely.
        """
        acquired, token, _ = await self._acquire_token(job, ttl_seconds)
        if acquired:
            self._last_tokens = getattr(self, "_last_tokens", {})
            self._last_tokens[job] = token
        return acquired

    async def _compare_and(self, job: str, token: str, *, renew_ms: int | None) -> bool:
        """Delete (renew_ms=None) or extend the lock only if ``token`` owns it."""
        key = self._key(job)
        client = await self._client()
        async with client.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(key)
                current = await pipe.get(key)
                value = current.decode() if isinstance(current, bytes) else current
                if value != token:
                    await cast(Callable[[], Awaitable[None]], pipe.unwatch)()
                    return False
                cast(Callable[[], None], pipe.multi)()
                if renew_ms is None:
                    pipe.delete(key)
                else:
                    pipe.pexpire(key, renew_ms)
                await pipe.execute()
                return True
            except WatchError:
                return False

    async def renew(self, job: str, token: str, ttl_seconds: int) -> bool:
        try:
            return await self._compare_and(job, token, renew_ms=int(ttl_seconds * 1000))
        except Exception as exc:
            logger.warning("leader lock renewal failed for %s: %s", job, type(exc).__name__)
            return False

    async def release(self, job: str, token: str | None = None) -> None:
        """Release only if we still own it (never free a newer leader)."""
        token = token or getattr(self, "_last_tokens", {}).get(job)
        if not token:
            return
        try:
            await self._compare_and(job, token, renew_ms=None)
        except Exception as exc:
            logger.debug("leader lock release skipped: %s", type(exc).__name__)

    @contextlib.asynccontextmanager
    async def hold(self, job: str, ttl_seconds: int) -> AsyncIterator[LeaderLease]:
        """Acquire, renew every ttl/3 while the body runs, release on exit.

        Yields a :class:`LeaderLease`; check ``lease.acquired`` before working
        and ``lease.is_lost`` between units of work.
        """
        acquired, token, coordinated = await self._acquire_token(job, ttl_seconds)
        lease = LeaderLease(job=job, token=token, acquired=acquired, coordinated=coordinated)
        if not acquired:
            yield lease
            return

        async def _renew_loop() -> None:
            interval = max(1.0, ttl_seconds / 3)
            while True:
                await asyncio.sleep(interval)
                if not await self.renew(job, token, ttl_seconds):
                    logger.error("leader lease for %s lost; stopping at next checkpoint", job)
                    lease.lost.set()
                    return

        renewer = asyncio.create_task(_renew_loop()) if coordinated else None
        try:
            yield lease
        finally:
            if renewer is not None:
                renewer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await renewer
            if coordinated:
                await self.release(job, token)
