from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from scripts.feature_adapter import cache_clip_manifest_features as cache_script
from ttvr.gpu_lock import DEFAULT_GPU_LOCK_PATH, exclusive_gpu_lock


def test_cache_cli_uses_shared_gpu_lock_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cache_clip_manifest_features.py",
            "--root",
            "data",
            "--samples",
            "samples.jsonl",
            "--taxa",
            "taxa.jsonl",
            "--cache-dir",
            "cache",
            "--model-cache-dir",
            "models",
        ],
    )

    args = cache_script._parse_args()

    assert args.gpu_lock == DEFAULT_GPU_LOCK_PATH
    assert args.no_gpu_lock is False


def test_cache_main_does_not_encode_when_shared_lock_is_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "gpu.lock"
    args = argparse.Namespace(no_gpu_lock=False, gpu_lock=lock_path)
    monkeypatch.setattr(cache_script, "_parse_args", lambda: args)
    monkeypatch.setattr(
        cache_script,
        "cache_manifest_features",
        lambda args: (_ for _ in ()).throw(AssertionError("must not encode")),
    )

    with exclusive_gpu_lock(lock_path, purpose="formal matrix"):
        with pytest.raises(SystemExit, match="feature caching did not start"):
            cache_script.main()


def test_cache_lock_opt_out_is_explicit_and_skips_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lock_path = tmp_path / "gpu.lock"
    args = argparse.Namespace(no_gpu_lock=True, gpu_lock=lock_path)
    monkeypatch.setattr(cache_script, "_parse_args", lambda: args)
    monkeypatch.setattr(
        cache_script,
        "cache_manifest_features",
        lambda args: [{"split": "train", "samples": 1}],
    )

    with exclusive_gpu_lock(lock_path, purpose="formal matrix"):
        cache_script.main()

    assert '"samples": 1' in capsys.readouterr().out
