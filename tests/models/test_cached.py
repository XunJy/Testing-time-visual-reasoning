from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from ttvr.models.cached import CachedFeatureBackend, normalise_features


class _CountingBackend(CachedFeatureBackend):
    def __init__(self, cache_identity: str) -> None:
        super().__init__(
            model_name="tiny-image-encoder",
            cache_identity=cache_identity,
            device=torch.device("cpu"),
            precision="fp32",
            text_batch_size=8,
            feature_dtype_name="torch.float32",
        )
        self.encoded_rows = 0

    def _encode_text_batch(self, texts):
        return normalise_features(torch.ones((len(texts), 4), dtype=torch.float32))

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        self.encoded_rows += images.shape[0]
        return normalise_features(images.float())


def _dataset() -> TensorDataset:
    images = torch.arange(1, 29, dtype=torch.float32).reshape(7, 4)
    labels = torch.tensor([0, 1, 0, 2, 1, 2, 0], dtype=torch.long)
    return TensorDataset(images, labels)


def test_resumable_image_cache_reuses_complete_shards(tmp_path: Path) -> None:
    cache_path = tmp_path / "features.pt"
    first = _CountingBackend("checkpoint-a")
    expected = first.encode_dataset(
        _dataset(),
        batch_size=2,
        cache_path=cache_path,
        cache_tag="locked-manifest",
        cache_shard_size=3,
    )

    assert first.encoded_rows == 7
    assert cache_path.is_file()
    assert len(list((tmp_path / "features.pt.shards").glob("part-*.pt"))) == 3

    cache_path.unlink()
    resumed = _CountingBackend("checkpoint-a")
    actual = resumed.encode_dataset(
        _dataset(),
        batch_size=2,
        cache_path=cache_path,
        cache_tag="locked-manifest",
        cache_shard_size=3,
    )

    assert resumed.encoded_rows == 0
    assert torch.equal(actual.features, expected.features)
    assert torch.equal(actual.labels, expected.labels)
    assert torch.equal(actual.sample_indices, torch.arange(7))


def test_resumable_image_cache_reencodes_only_a_corrupt_shard(tmp_path: Path) -> None:
    cache_path = tmp_path / "features.pt"
    original = _CountingBackend("checkpoint-a")
    original.encode_dataset(
        _dataset(),
        batch_size=2,
        cache_path=cache_path,
        cache_tag="locked-manifest",
        cache_shard_size=3,
    )
    cache_path.unlink()
    middle = tmp_path / "features.pt.shards" / "part-000000003-000000006.pt"
    middle.write_bytes(b"not a torch cache")

    resumed = _CountingBackend("checkpoint-a")
    result = resumed.encode_dataset(
        _dataset(),
        batch_size=2,
        cache_path=cache_path,
        cache_tag="locked-manifest",
        cache_shard_size=3,
    )

    assert resumed.encoded_rows == 3
    assert result.size == 7


def test_resumable_image_cache_rejects_another_checkpoint(tmp_path: Path) -> None:
    cache_path = tmp_path / "features.pt"
    first = _CountingBackend("checkpoint-a")
    first.encode_dataset(
        _dataset(),
        batch_size=2,
        cache_path=cache_path,
        cache_tag="locked-manifest",
        cache_shard_size=3,
    )
    cache_path.unlink()

    different = _CountingBackend("checkpoint-b")
    different.encode_dataset(
        _dataset(),
        batch_size=2,
        cache_path=cache_path,
        cache_tag="locked-manifest",
        cache_shard_size=3,
    )

    assert different.encoded_rows == 7


def test_cache_shards_require_a_destination(tmp_path: Path) -> None:
    backend = _CountingBackend("checkpoint-a")

    with pytest.raises(ValueError, match="requires cache_path"):
        backend.encode_dataset(
            _dataset(),
            batch_size=2,
            cache_shard_size=3,
        )

    with pytest.raises(ValueError, match="positive"):
        backend.encode_dataset(
            _dataset(),
            batch_size=2,
            cache_path=tmp_path / "features.pt",
            cache_shard_size=0,
        )
