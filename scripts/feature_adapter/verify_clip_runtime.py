#!/usr/bin/env python3
"""Verify the pinned OpenAI CLIP runtime without loading the model onto a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ttvr.models.cached import validate_text_cache_file
from ttvr.models.clip import (
    DEFAULT_CLIP_MODEL,
    OPENAI_CLIP_COMMIT,
    VIT_L14_336_CHECKPOINT_FILENAME,
    VIT_L14_336_CHECKPOINT_SHA256,
    verify_openai_clip_checkpoint,
    verify_openai_clip_installation,
)

_CLIP_FEATURE_DIM = 768


def verify_runtime(
    model_cache_dir: Path | str,
    text_cache: Path | str,
) -> dict[str, Any]:
    """Return the audited code, weight, and optional text-cache identity."""

    installation = verify_openai_clip_installation(expected_commit=OPENAI_CLIP_COMMIT)
    checkpoint = verify_openai_clip_checkpoint(
        model_cache_dir,
        model_name=DEFAULT_CLIP_MODEL,
        checkpoint_filename=VIT_L14_336_CHECKPOINT_FILENAME,
        expected_sha256=VIT_L14_336_CHECKPOINT_SHA256,
    )
    cache_identity = f"openai-clip:{DEFAULT_CLIP_MODEL}@{installation.commit_id}"
    text_path = Path(text_cache).expanduser()
    text_cache_keys = None
    if text_path.exists():
        text_cache_keys = validate_text_cache_file(
            text_path,
            cache_identity=cache_identity,
            model_name=DEFAULT_CLIP_MODEL,
            precision="fp32",
            dtype_name="torch.float32",
            feature_dim=_CLIP_FEATURE_DIM,
        )
    return {
        "cache_identity": cache_identity,
        "clip_distribution": installation.distribution,
        "clip_version": installation.version,
        "clip_repository_url": installation.repository_url,
        "clip_commit": installation.commit_id,
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_size_bytes": checkpoint.size_bytes,
        "text_cache_path": str(text_path.resolve()),
        "text_cache_keys": text_cache_keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_runtime(args.model_cache_dir, args.text_cache),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
