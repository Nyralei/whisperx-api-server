"""Backend-agnostic job processing lease.

The claims/{job_id} object makes a redelivered copy of an in-flight job defer
instead of starting a concurrent duplicate run, and it carries the
delivery-attempt counter used for DLQ routing. Acquisition needs an atomic
create-if-absent, which every backend proves at startup.
"""

from __future__ import annotations

import json
import time

from .contracts import ObjectStore


class LeaseManager:
    def __init__(self, store: ObjectStore, *, prefix: str = "claims/") -> None:
        self._store = store
        self._prefix = prefix

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"

    async def read(self, job_id: str) -> dict | None:
        data = await self._store.get_bytes(key=self._key(job_id))
        if data is None:
            return None
        try:
            lease = json.loads(data)
            if isinstance(lease, dict):
                return lease
        except ValueError:
            pass
        # Legacy bare-int claim counter from a pre-lease worker: preserve the
        # attempt count, treat as expired so a lease-aware worker can take over.
        try:
            attempts = int(data.decode().strip() or "0")
        except ValueError:
            attempts = 0
        return {"attempts": attempts, "owner": None, "expires_at": 0.0}

    async def _put_if_absent(self, job_id: str, lease: dict) -> bool:
        return await self._store.put_if_absent(
            key=self._key(job_id), data=json.dumps(lease).encode()
        )

    async def acquire(
        self, job_id: str, worker_id: str, ttl_seconds: float
    ) -> tuple[bool, int]:
        """Claim exclusive processing rights for a job; return (acquired, attempts).

        A live lease held by another worker means the job is being processed right
        now (e.g. a rebalance redelivered its uncommitted message) — the caller must
        defer, not process. An expired lease or one left by a previous incarnation
        of this worker (same worker_id after a crash-restart) is taken over, which
        also advances the attempts counter used for DLQ routing.
        """
        fresh = {
            "attempts": 1,
            "owner": worker_id,
            "expires_at": time.time() + ttl_seconds,
        }
        if await self._put_if_absent(job_id, fresh):
            return True, 1

        current = await self.read(job_id)
        if current is None:
            # Deleted between the failed create and the read (holder just released);
            # one retry, then give up and let the requeued copy sort it out.
            if await self._put_if_absent(job_id, fresh):
                return True, 1
            current = await self.read(job_id) or {"attempts": 0}

        attempts = int(current.get("attempts") or 0)
        is_live = (
            float(current.get("expires_at") or 0.0) > time.time()
            and current.get("owner") != worker_id
        )
        if is_live:
            return False, attempts

        takeover = {
            "attempts": attempts + 1,
            "owner": worker_id,
            "expires_at": time.time() + ttl_seconds,
        }
        await self.release(job_id)
        if await self._put_if_absent(job_id, takeover):
            return True, attempts + 1
        return False, attempts

    async def renew(self, job_id: str, worker_id: str, ttl_seconds: float) -> bool:
        """Extend a held lease; False if it is gone or owned by someone else."""
        current = await self.read(job_id)
        if current is None or current.get("owner") != worker_id:
            return False
        current["expires_at"] = time.time() + ttl_seconds
        await self._store.put_bytes(
            key=self._key(job_id), data=json.dumps(current).encode()
        )
        return True

    async def release(self, job_id: str) -> None:
        await self._store.delete(key=self._key(job_id))
