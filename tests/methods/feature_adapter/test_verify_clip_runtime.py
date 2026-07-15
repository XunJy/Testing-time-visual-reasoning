from __future__ import annotations

from pathlib import Path

from scripts.feature_adapter import verify_clip_runtime as verifier
from ttvr.models.clip import (
    OPENAI_CLIP_COMMIT,
    OPENAI_CLIP_REPOSITORY,
    VIT_L14_336_CHECKPOINT_SHA256,
    OpenAIClipCheckpoint,
    OpenAIClipInstallation,
)


def test_runtime_verifier_uses_verified_commit_for_text_cache_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    text_cache = tmp_path / "text.pt"
    text_cache.write_bytes(b"test double")
    installation = OpenAIClipInstallation(
        distribution="clip",
        version="1.0",
        repository_url=OPENAI_CLIP_REPOSITORY,
        vcs="git",
        commit_id=OPENAI_CLIP_COMMIT,
    )
    checkpoint = OpenAIClipCheckpoint(
        model_name="ViT-L/14@336px",
        path=tmp_path / "ViT-L-14-336px.pt",
        sha256=VIT_L14_336_CHECKPOINT_SHA256,
        size_bytes=123,
    )
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        verifier,
        "verify_openai_clip_installation",
        lambda **kwargs: installation,
    )
    monkeypatch.setattr(
        verifier,
        "verify_openai_clip_checkpoint",
        lambda *args, **kwargs: checkpoint,
    )

    def validate(path: Path, **kwargs: object) -> int:
        calls["path"] = path
        calls.update(kwargs)
        return 17

    monkeypatch.setattr(verifier, "validate_text_cache_file", validate)

    identity = verifier.verify_runtime(tmp_path, text_cache)

    expected_cache_identity = f"openai-clip:ViT-L/14@336px@{OPENAI_CLIP_COMMIT}"
    assert calls["cache_identity"] == expected_cache_identity
    assert calls["feature_dim"] == 768
    assert identity["cache_identity"] == expected_cache_identity
    assert identity["checkpoint_sha256"] == VIT_L14_336_CHECKPOINT_SHA256
    assert identity["text_cache_keys"] == 17
