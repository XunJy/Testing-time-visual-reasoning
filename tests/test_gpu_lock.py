from __future__ import annotations

from pathlib import Path

import pytest

from ttvr.gpu_lock import GPULockBusyError, exclusive_gpu_lock


def test_exclusive_gpu_lock_fails_fast_and_releases(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"

    with exclusive_gpu_lock(lock_path, purpose="first job"):
        with pytest.raises(GPULockBusyError, match="second job did not start"):
            with exclusive_gpu_lock(lock_path, purpose="second job"):
                pass

    with exclusive_gpu_lock(lock_path, purpose="third job"):
        pass


def test_exclusive_gpu_lock_requires_a_purpose(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="purpose"):
        with exclusive_gpu_lock(tmp_path / "gpu.lock", purpose=" "):
            pass
