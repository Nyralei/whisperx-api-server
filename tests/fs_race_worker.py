"""Child-process worker for the cross-process atomic-create race test.

Kept in its own module so `spawn` re-imports something tiny rather than a test
module with fixtures and side effects.
"""

from __future__ import annotations

import os


def try_exclusive_create(barrier, target: str) -> bool:
    """Race every caller onto one O_CREAT|O_EXCL create; True if this one won."""
    barrier.wait()
    try:
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    os.close(fd)
    return True


def run(barrier, target: str, results) -> None:
    results.append(try_exclusive_create(barrier, target))
