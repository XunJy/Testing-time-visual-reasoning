from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ttvr.models.cached import (
    CachedFeatureBackend,
    normalise_features,
    torch_load_tensors,
    validate_text_cache_file,
)
from ttvr.models.open_clip import (
    EVA02_CLIP_L14_336,
    OpenCLIPBackend,
    _verified_hf_checkpoint,
)


def test_eva02_checkpoint_identity_is_fully_locked() -> None:
    checkpoint = EVA02_CLIP_L14_336

    assert checkpoint.model_name == "EVA02-L-14-336"
    assert checkpoint.pretrained_tag == "merged2b_s6b_b61k"
    assert checkpoint.hf_repo_id == ("timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k")
    assert checkpoint.hf_revision == "4f62907359c8506be7021582f360564693b22c15"
    assert checkpoint.checkpoint_filename == "open_clip_model.safetensors"
    assert checkpoint.checkpoint_sha256 == (
        "f753bca0e8327f77e8845b0af2510d599c3e4614237007b48078c791f2cf391c"
    )
    assert checkpoint.checkpoint_bytes == 856_239_456
    assert checkpoint.context_length == 77
    assert checkpoint.image_size == 336
    assert checkpoint.interpolation == "bicubic"
    assert checkpoint.resize_mode == "shortest"
    assert checkpoint.crop_mode == "center"


def test_checkpoint_cache_identity_changes_with_revision_and_tokenizer() -> None:
    checkpoint = EVA02_CLIP_L14_336

    changed_revision = replace(checkpoint, hf_revision="0" * 40)
    changed_tokenizer = replace(checkpoint, tokenizer="another-tokenizer")

    assert checkpoint.cache_identity != changed_revision.cache_identity
    assert checkpoint.cache_identity != changed_tokenizer.cache_identity


def test_hf_download_uses_exact_revision_and_verifies_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"locked-safetensors-test"
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(payload)
    checkpoint = replace(
        EVA02_CLIP_L14_336,
        checkpoint_bytes=len(payload),
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
    )
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(model_file)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download),
    )

    result = _verified_hf_checkpoint(checkpoint, tmp_path / "cache")

    assert result == model_file
    assert calls == [
        {
            "repo_id": checkpoint.hf_repo_id,
            "filename": checkpoint.checkpoint_filename,
            "revision": checkpoint.hf_revision,
            "cache_dir": str(tmp_path / "cache"),
        }
    ]


def test_hf_download_rejects_wrong_checkpoint_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"wrong-contents")
    checkpoint = replace(
        EVA02_CLIP_L14_336,
        checkpoint_bytes=model_file.stat().st_size,
        checkpoint_sha256="0" * 64,
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda **_: str(model_file)),
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _verified_hf_checkpoint(checkpoint, None)


class _FakeOpenCLIPModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        values = torch.arange(tokens.shape[0] * 4, dtype=torch.float16).reshape(-1, 4)
        return values + 1

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(images.shape[0] * 4, dtype=torch.float16).reshape(-1, 4)
        return rows + 1


def test_open_clip_backend_locks_factory_and_converts_features_to_fp32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    checkpoint_path.touch()
    calls: dict[str, object] = {}
    model = _FakeOpenCLIPModel()

    def fake_create(model_name, **kwargs):
        calls["model_name"] = model_name
        calls["create_kwargs"] = kwargs
        return model, object(), "locked-preprocess"

    def fake_get_tokenizer(model_name, **kwargs):
        calls["tokenizer_model"] = model_name
        calls["tokenizer_kwargs"] = kwargs
        return lambda texts: torch.ones((len(texts), 77), dtype=torch.long)

    monkeypatch.setattr(
        "ttvr.models.open_clip._verified_hf_checkpoint",
        lambda checkpoint, cache_dir: checkpoint_path,
    )
    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        SimpleNamespace(
            create_model_and_transforms=fake_create,
            get_tokenizer=fake_get_tokenizer,
        ),
    )

    backend = OpenCLIPBackend(device="cpu", precision="fp32", model_cache_dir=tmp_path)
    text_features = backend._encode_text_batch(("one", "two"))
    image_features = backend.encode_images(torch.ones((2, 3, 336, 336)))

    assert backend.preprocess == "locked-preprocess"
    assert backend.feature_dtype_name == "torch.float32"
    assert text_features.dtype == torch.float32
    assert image_features.dtype == torch.float32
    assert torch.allclose(text_features.norm(dim=-1), torch.ones(2))
    assert torch.allclose(image_features.norm(dim=-1), torch.ones(2))
    assert calls["model_name"] == EVA02_CLIP_L14_336.model_name
    assert calls["tokenizer_model"] == EVA02_CLIP_L14_336.model_name
    create_kwargs = calls["create_kwargs"]
    assert isinstance(create_kwargs, dict)
    assert create_kwargs["pretrained"] == str(checkpoint_path)
    assert create_kwargs["precision"] == "fp32"
    assert create_kwargs["device"] == torch.device("cpu")
    assert create_kwargs["weights_only"] is True
    assert create_kwargs["force_image_size"] == 336
    assert create_kwargs["image_mean"] == EVA02_CLIP_L14_336.image_mean
    assert create_kwargs["image_std"] == EVA02_CLIP_L14_336.image_std
    assert create_kwargs["image_interpolation"] == "bicubic"
    assert create_kwargs["image_resize_mode"] == "shortest"
    assert calls["tokenizer_kwargs"] == {
        "context_length": 77,
        "cache_dir": str(tmp_path),
    }


class _TinyCachedBackend(CachedFeatureBackend):
    def __init__(self, cache_identity: str, cache_path: Path | None = None) -> None:
        super().__init__(
            model_name="same-architecture",
            cache_identity=cache_identity,
            device=torch.device("cpu"),
            precision="fp32",
            text_batch_size=8,
            feature_dtype_name="torch.float32",
            feature_dim=3,
            text_cache_path=cache_path,
        )

    def _encode_text_batch(self, texts):
        features = torch.arange(len(texts) * 3, dtype=torch.float32).reshape(-1, 3) + 1
        return normalise_features(features)


def test_text_cache_does_not_cross_checkpoint_identity(tmp_path: Path) -> None:
    cache_path = tmp_path / "text.pt"
    first = _TinyCachedBackend("checkpoint-a", cache_path)
    first.warm_text_cache(("alpha", "beta"))
    first.save_text_cache()

    matching = _TinyCachedBackend("checkpoint-a", cache_path)
    different = _TinyCachedBackend("checkpoint-b", cache_path)

    assert matching.text_cache_size == 2
    assert different.text_cache_size == 0


def test_immutable_preflight_rejects_text_cache_identity_mismatch(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "text.pt"
    original = _TinyCachedBackend("checkpoint-a", cache_path)
    original.warm_text_cache(("alpha", "beta"))
    original.save_text_cache()

    assert (
        validate_text_cache_file(
            cache_path,
            cache_identity="checkpoint-a",
            model_name="same-architecture",
            precision="fp32",
            dtype_name="torch.float32",
            feature_dim=3,
        )
        == 2
    )
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        validate_text_cache_file(
            cache_path,
            cache_identity="checkpoint-b",
            model_name="same-architecture",
            precision="fp32",
            dtype_name="torch.float32",
            feature_dim=3,
        )


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("wrong-dimension", "dimension mismatch"),
        ("non-finite", "Non-finite"),
        ("non-unit", "Non-unit"),
        ("duplicate-location", "one-to-one"),
        ("orphan-row", "one-to-one"),
    ],
)
def test_matching_text_cache_fails_closed_on_feature_or_alignment_damage(
    tmp_path: Path,
    damage: str,
    message: str,
) -> None:
    cache_path = tmp_path / f"{damage}.pt"
    original = _TinyCachedBackend("checkpoint-a", cache_path)
    original.warm_text_cache(("alpha", "beta"))
    original.save_text_cache()
    payload = torch_load_tensors(cache_path)

    if damage == "wrong-dimension":
        payload["blocks"][0] = normalise_features(torch.ones((2, 4)))
    elif damage == "non-finite":
        payload["blocks"][0][0, 0] = float("nan")
    elif damage == "non-unit":
        payload["blocks"][0][0].mul_(0.5)
    elif damage == "duplicate-location":
        payload["locations"]["beta"] = payload["locations"]["alpha"]
    elif damage == "orphan-row":
        del payload["locations"]["beta"]
    else:  # pragma: no cover - guarded by the parametrization above.
        raise AssertionError(damage)
    torch.save(payload, cache_path)

    with pytest.raises(RuntimeError, match=message):
        _TinyCachedBackend("checkpoint-a", cache_path)


def test_matching_text_cache_fails_closed_on_serialization_damage(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "text.pt"
    cache_path.write_bytes(b"not a torch payload")

    with pytest.raises(RuntimeError, match="Cannot load text cache"):
        _TinyCachedBackend("checkpoint-a", cache_path)
