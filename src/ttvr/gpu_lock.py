"""Shared fail-fast lock for mutually exclusive GPU experiment work."""

from __future__ import annotations

import contextlib
import fcntl
from collections.abc import Iterator
from pathlib import Path

DEFAULT_GPU_LOCK_PATH = Path("/tmp/ttvr-gpu.lock")


class GPULockBusyError(RuntimeError):
    """Another cooperating cache or experiment process already owns the GPU."""


@contextlib.contextmanager
def exclusive_gpu_lock(
    path: Path | str = DEFAULT_GPU_LOCK_PATH,
    *,
    purpose: str,
) -> Iterator[None]:
    """Acquire the shared GPU lock or fail immediately without doing GPU work."""

    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("purpose must be a non-empty string")
    lock_path = Path(path).expanduser()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GPULockBusyError(
                f"GPU lock is busy; {purpose} did not start: {lock_path}"
            ) from error
        acquired = True
        yield
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
