"""Shared deterministic feature caching for dual-encoder backends.

This module owns mechanics that are independent of a particular model family:
batched text caching, prompt pooling, and deterministic dataset encoding.  A
backend only has to implement ``_encode_text_batch`` and ``encode_images``.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from .base import ImageFeatureSet, ProgressCallback, TextFeatureTable

_TEXT_CACHE_FORMAT = 2
_IMAGE_CACHE_FORMAT = 2
_IMAGE_SHARD_FORMAT = 1


def _load_text_cache_payload(path: Path) -> Any:
    """Load one text cache and turn serialization damage into a clear failure."""

    try:
        return torch_load_tensors(path)
    except Exception as error:
        raise RuntimeError(f"Cannot load text cache: {path}: {error}") from error


def _validated_text_cache_payload(
    payload: Any,
    *,
    cache_path: Path,
    cache_identity: str,
    model_name: str,
    precision: str,
    dtype_name: str,
    feature_dim: int | None,
    allow_metadata_mismatch: bool,
) -> tuple[list[Tensor], dict[str, tuple[int, int]]] | None:
    """Deep-validate a persistent text cache without trusting its metadata."""

    if not isinstance(payload, dict) or payload.get("format") != _TEXT_CACHE_FORMAT:
        raise RuntimeError(f"Invalid text cache payload: {cache_path}")
    metadata_fields = {
        "cache_identity": cache_identity,
        "model_name": model_name,
        "precision": precision,
        "dtype": dtype_name,
    }
    metadata_matches = all(payload.get(key) == value for key, value in metadata_fields.items())
    if not metadata_matches:
        if allow_metadata_mismatch:
            return None
        mismatches = {
            key: {"expected": expected, "found": payload.get(key)}
            for key, expected in metadata_fields.items()
            if payload.get(key) != expected
        }
        raise RuntimeError(f"Text cache metadata mismatch in {cache_path}: {mismatches}")

    blocks = payload.get("blocks")
    locations = payload.get("locations")
    if not isinstance(blocks, list):
        raise RuntimeError(f"Invalid tensor blocks in text cache: {cache_path}")
    if not isinstance(locations, dict):
        raise RuntimeError(f"Invalid locations in text cache: {cache_path}")

    parsed_blocks: list[Tensor] = []
    inferred_dim = feature_dim
    for block_index, block in enumerate(blocks):
        if not (
            isinstance(block, Tensor)
            and block.layout == torch.strided
            and block.ndim == 2
            and block.shape[0] > 0
            and block.shape[1] > 0
            and block.is_floating_point()
            and str(block.dtype) == dtype_name
        ):
            raise RuntimeError(f"Invalid tensor block {block_index} in text cache: {cache_path}")
        if inferred_dim is None:
            inferred_dim = block.shape[1]
        if block.shape[1] != inferred_dim:
            raise RuntimeError(
                f"Text feature dimension mismatch in {cache_path}: "
                f"expected {inferred_dim}, found {block.shape[1]} in block {block_index}"
            )
        if not bool(torch.isfinite(block).all()):
            raise RuntimeError(f"Non-finite text features in {cache_path}, block {block_index}")
        norms = torch.linalg.vector_norm(block.detach().float(), dim=1)
        if not torch.allclose(
            norms,
            torch.ones_like(norms),
            rtol=1e-3,
            atol=1e-3,
        ):
            raise RuntimeError(f"Non-unit text features in {cache_path}, block {block_index}")
        parsed_blocks.append(block.detach().cpu().contiguous())

    parsed_locations: dict[str, tuple[int, int]] = {}
    for text, location in locations.items():
        if not (
            isinstance(text, str)
            and text.strip()
            and isinstance(location, (list, tuple))
            and len(location) == 2
            and all(type(value) is int for value in location)
        ):
            raise RuntimeError(f"Invalid location entry in {cache_path}")
        block_index, row_index = location
        if not (
            0 <= block_index < len(parsed_blocks)
            and 0 <= row_index < parsed_blocks[block_index].shape[0]
        ):
            raise RuntimeError(f"Out-of-range location in {cache_path}")
        parsed_locations[text] = (block_index, row_index)

    expected_locations = {
        (block_index, row_index)
        for block_index, block in enumerate(parsed_blocks)
        for row_index in range(block.shape[0])
    }
    actual_locations = set(parsed_locations.values())
    if len(actual_locations) != len(parsed_locations) or actual_locations != expected_locations:
        raise RuntimeError(f"Text cache keys and tensor rows are not one-to-one in {cache_path}")
    return parsed_blocks, parsed_locations


def validate_text_cache_file(
    path: Path | str,
    *,
    cache_identity: str,
    model_name: str,
    precision: str,
    dtype_name: str,
    feature_dim: int | None = None,
) -> int:
    """Fail closed unless an existing text cache exactly matches and is sound.

    Returns the number of validated prompt keys. This stricter entry point is
    intended for immutable experiment preflights; backends may still ignore a
    structurally valid cache belonging to a different model identity.
    """

    cache_path = Path(path).expanduser()
    if not cache_path.is_file():
        raise RuntimeError(f"Missing text cache: {cache_path}")
    result = _validated_text_cache_payload(
        _load_text_cache_payload(cache_path),
        cache_path=cache_path,
        cache_identity=cache_identity,
        model_name=model_name,
        precision=precision,
        dtype_name=dtype_name,
        feature_dim=feature_dim,
        allow_metadata_mismatch=False,
    )
    assert result is not None
    return len(result[1])


def torch_load_tensors(path: Path) -> Any:
    """Load tensor-only caches safely across supported PyTorch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 does not expose weights_only.
        return torch.load(path, map_location="cpu")


def atomic_torch_save(value: Any, path: Path) -> None:
    """Atomically replace a cache file without leaving partial tensors behind."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def normalise_features(features: Tensor) -> Tensor:
    """L2-normalise the final feature dimension."""

    return functional.normalize(features, dim=-1)


def _validated_image_feature_payload(
    payload: Any,
    *,
    cache_identity: str,
    model_name: str,
    precision: str,
    dtype_name: str,
    cache_tag: str,
    sample_count: int,
) -> ImageFeatureSet | None:
    """Return a complete aligned cache only when every locked field matches."""

    if not isinstance(payload, dict):
        return None
    metadata_matches = (
        payload.get("format") == _IMAGE_CACHE_FORMAT
        and payload.get("cache_identity") == cache_identity
        and payload.get("model_name") == model_name
        and payload.get("precision") == precision
        and payload.get("dtype") == dtype_name
        and payload.get("cache_tag") == cache_tag
        and payload.get("sample_count") == sample_count
    )
    features = payload.get("features")
    labels = payload.get("labels")
    sample_indices = payload.get("sample_indices")
    if not metadata_matches or not all(
        isinstance(value, Tensor) for value in (features, labels, sample_indices)
    ):
        return None
    assert isinstance(features, Tensor)
    assert isinstance(labels, Tensor)
    assert isinstance(sample_indices, Tensor)
    if (
        features.ndim != 2
        or features.shape[0] != sample_count
        or labels.shape != (sample_count,)
        or sample_indices.shape != (sample_count,)
        or not features.is_floating_point()
        or not torch.equal(
            sample_indices.detach().cpu().long(),
            torch.arange(sample_count, dtype=torch.long),
        )
    ):
        return None
    return ImageFeatureSet(
        features=features.detach().cpu().contiguous(),
        labels=labels.detach().cpu().long().contiguous(),
        sample_indices=sample_indices.detach().cpu().long().contiguous(),
    )


def _validated_image_shard_payload(
    payload: Any,
    *,
    cache_identity: str,
    model_name: str,
    precision: str,
    dtype_name: str,
    cache_tag: str,
    sample_count: int,
    start: int,
    stop: int,
) -> ImageFeatureSet | None:
    """Validate one resumable image-feature shard against its parent cache."""

    if not isinstance(payload, dict):
        return None
    shard_count = stop - start
    metadata_matches = (
        payload.get("format") == _IMAGE_SHARD_FORMAT
        and payload.get("cache_identity") == cache_identity
        and payload.get("model_name") == model_name
        and payload.get("precision") == precision
        and payload.get("dtype") == dtype_name
        and payload.get("cache_tag") == cache_tag
        and payload.get("sample_count") == sample_count
        and payload.get("shard_start") == start
        and payload.get("shard_stop") == stop
    )
    features = payload.get("features")
    labels = payload.get("labels")
    sample_indices = payload.get("sample_indices")
    if not metadata_matches or not all(
        isinstance(value, Tensor) for value in (features, labels, sample_indices)
    ):
        return None
    assert isinstance(features, Tensor)
    assert isinstance(labels, Tensor)
    assert isinstance(sample_indices, Tensor)
    if (
        features.ndim != 2
        or features.shape[0] != shard_count
        or labels.shape != (shard_count,)
        or sample_indices.shape != (shard_count,)
        or not features.is_floating_point()
        or not torch.equal(
            sample_indices.detach().cpu().long(),
            torch.arange(start, stop, dtype=torch.long),
        )
    ):
        return None
    return ImageFeatureSet(
        features=features.detach().cpu().contiguous(),
        labels=labels.detach().cpu().long().contiguous(),
        sample_indices=sample_indices.detach().cpu().long().contiguous(),
    )


class CachedFeatureBackend:
    """Common cache and aggregation implementation for dual encoders."""

    def __init__(
        self,
        *,
        model_name: str,
        cache_identity: str,
        device: torch.device,
        precision: str,
        text_batch_size: int,
        feature_dtype_name: str,
        feature_dim: int | None = None,
        text_cache_path: Path | str | None = None,
    ) -> None:
        if text_batch_size <= 0:
            raise ValueError("text_batch_size must be positive")
        if not model_name.strip() or not cache_identity.strip():
            raise ValueError("model_name and cache_identity must not be empty")
        if feature_dim is not None and feature_dim <= 0:
            raise ValueError("feature_dim must be positive when provided")
        self.model_name = model_name
        self.cache_identity = cache_identity
        self.device = device
        self.precision = precision
        self.text_batch_size = text_batch_size
        self._dtype_name = feature_dtype_name
        self._feature_dim = feature_dim

        # Each block is contiguous; locations avoid 150k+ independent tensor
        # storages when the complete CUB prompt cache is serialised.
        self._text_blocks: list[Tensor] = []
        self._text_locations: dict[str, tuple[int, int]] = {}
        self._text_cache_path: Path | None = None
        if text_cache_path is not None:
            self.configure_text_cache(text_cache_path)

    @staticmethod
    def resolve_device(device: str | torch.device | None) -> torch.device:
        """Resolve an optional device and fail closed on unavailable CUDA."""

        if device is not None:
            resolved = torch.device(device)
        elif torch.cuda.is_available():
            resolved = torch.device("cuda")
        else:
            resolved = torch.device("cpu")
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return resolved

    @property
    def text_cache_size(self) -> int:
        """Number of unique prompt strings currently cached."""

        return len(self._text_locations)

    @property
    def feature_dtype_name(self) -> str:
        """Feature dtype used for similarities and prompt aggregation."""

        return self._dtype_name

    def configure_text_cache(self, path: Path | str) -> None:
        """Select and, when present, load a persistent text-feature cache."""

        cache_path = Path(path).expanduser()
        if self._text_cache_path == cache_path:
            return
        if self._text_locations:
            raise RuntimeError("Configure the persistent text cache before encoding prompts")
        self._text_cache_path = cache_path
        if not cache_path.is_file():
            return

        result = _validated_text_cache_payload(
            _load_text_cache_payload(cache_path),
            cache_path=cache_path,
            cache_identity=self.cache_identity,
            model_name=self.model_name,
            precision=self.precision,
            dtype_name=self._dtype_name,
            feature_dim=self._feature_dim,
            allow_metadata_mismatch=True,
        )
        if result is None:
            return
        self._text_blocks, self._text_locations = result

    def save_text_cache(self) -> Path | None:
        """Atomically persist all cached text features, if a path is configured."""

        if self._text_cache_path is None:
            return None
        payload = {
            "format": _TEXT_CACHE_FORMAT,
            "cache_identity": self.cache_identity,
            "model_name": self.model_name,
            "precision": self.precision,
            "dtype": self._dtype_name,
            "blocks": self._text_blocks,
            "locations": self._text_locations,
        }
        atomic_torch_save(payload, self._text_cache_path)
        return self._text_cache_path

    @staticmethod
    def _validated_texts(texts: Iterable[str]) -> tuple[str, ...]:
        result = tuple(texts)
        if not result:
            raise ValueError("At least one text is required")
        if any(not isinstance(text, str) or not text.strip() for text in result):
            raise ValueError("All prompts must be non-empty strings")
        return result

    def _encode_text_batch(self, texts: Sequence[str]) -> Tensor:
        raise NotImplementedError

    def warm_text_cache(
        self,
        texts: Iterable[str],
        *,
        progress: ProgressCallback | None = None,
    ) -> None:
        """Batch-encode any missing unique texts without materialising a table."""

        requested = self._validated_texts(texts)
        missing = tuple(
            text for text in dict.fromkeys(requested) if text not in self._text_locations
        )
        total = len(missing)
        if total == 0:
            if progress is not None:
                progress("text", 0, 0)
            return

        with torch.inference_mode():
            for start in range(0, total, self.text_batch_size):
                chunk = missing[start : start + self.text_batch_size]
                features = self._encode_text_batch(chunk)
                if features.ndim != 2 or features.shape[0] != len(chunk):
                    raise RuntimeError("Text encoder returned an invalid feature matrix")
                block = features.detach().cpu().contiguous()
                if self._feature_dim is None:
                    self._feature_dim = block.shape[1]
                validation = _validated_text_cache_payload(
                    {
                        "format": _TEXT_CACHE_FORMAT,
                        "cache_identity": self.cache_identity,
                        "model_name": self.model_name,
                        "precision": self.precision,
                        "dtype": self._dtype_name,
                        "blocks": [block],
                        "locations": {text: (0, row_index) for row_index, text in enumerate(chunk)},
                    },
                    cache_path=self._text_cache_path or Path("<in-memory-text-cache>"),
                    cache_identity=self.cache_identity,
                    model_name=self.model_name,
                    precision=self.precision,
                    dtype_name=self._dtype_name,
                    feature_dim=self._feature_dim,
                    allow_metadata_mismatch=False,
                )
                assert validation is not None
                block = validation[0][0]
                block_index = len(self._text_blocks)
                self._text_blocks.append(block)
                for row_index, text in enumerate(chunk):
                    self._text_locations[text] = (block_index, row_index)
                if progress is not None:
                    progress("text", min(start + len(chunk), total), total)

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        progress: ProgressCallback | None = None,
    ) -> Tensor:
        """Return normalised text features in input order, encoding in batches."""

        requested = self._validated_texts(texts)
        self.warm_text_cache(requested, progress=progress)
        rows = [
            self._text_blocks[block_index][row_index]
            for block_index, row_index in (self._text_locations[text] for text in requested)
        ]
        return torch.stack(rows).to(self.device, non_blocking=True)

    def build_text_feature_table(
        self,
        texts: Iterable[str],
        *,
        progress: ProgressCallback | None = None,
    ) -> TextFeatureTable:
        """Build one dense device table for a deterministic unique text list."""

        unique = tuple(dict.fromkeys(self._validated_texts(texts)))
        features = self.encode_texts(unique, progress=progress)
        return TextFeatureTable(
            texts=unique,
            features=features,
            index={text: index for index, text in enumerate(unique)},
        )

    def pool_prompt_groups(
        self,
        groups: Sequence[Sequence[str]],
        table: TextFeatureTable | None = None,
    ) -> Tensor:
        """Mean-pool normalised prompt features, then normalise each group."""

        if not groups or any(not group for group in groups):
            raise ValueError("Every prompt group must contain at least one text")
        if table is None:
            table = self.build_text_feature_table(text for group in groups for text in group)

        flat_indices: list[int] = []
        lengths: list[int] = []
        for group in groups:
            lengths.append(len(group))
            try:
                flat_indices.extend(table.index[text] for text in group)
            except KeyError as error:
                raise KeyError(f"Prompt is missing from text table: {error.args[0]}") from error

        indices = torch.tensor(flat_indices, dtype=torch.long, device=table.features.device)
        selected = table.features.index_select(0, indices)
        length_tensor = torch.tensor(lengths, dtype=torch.long, device=table.features.device)
        group_ids = torch.repeat_interleave(
            torch.arange(len(groups), device=table.features.device),
            length_tensor,
        )
        sums = torch.zeros(
            (len(groups), table.features.shape[1]),
            dtype=table.features.dtype,
            device=table.features.device,
        )
        sums.index_add_(0, group_ids, selected)
        means = sums / length_tensor.to(sums.dtype).unsqueeze(1)
        return normalise_features(means)

    def encode_images(self, images: Tensor) -> Tensor:
        raise NotImplementedError

    def encode_dataset(
        self,
        dataset: Dataset[Any],
        *,
        batch_size: int,
        num_workers: int = 0,
        max_samples: int | None = None,
        seed: int = 0,
        cache_path: Path | str | None = None,
        cache_tag: str = "",
        cache_shard_size: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> ImageFeatureSet:
        """Batch-encode a deterministic prefix, optionally with resumable shards.

        When ``cache_shard_size`` is set, every completed shard is saved
        atomically beside the final cache.  A later invocation validates and
        reuses those shards before encoding missing rows, then writes the
        ordinary format-2 cache expected by all downstream consumers.
        """

        if batch_size <= 0 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if cache_shard_size is not None and cache_shard_size <= 0:
            raise ValueError("cache_shard_size must be positive when provided")
        dataset_size = len(dataset)
        if dataset_size <= 0:
            raise ValueError("dataset must not be empty")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        sample_count = min(dataset_size, max_samples or dataset_size)
        resolved_cache = None if cache_path is None else Path(cache_path).expanduser()
        if cache_shard_size is not None and resolved_cache is None:
            raise ValueError("cache_shard_size requires cache_path")

        if resolved_cache is not None and resolved_cache.is_file():
            try:
                payload = torch_load_tensors(resolved_cache)
                cached = _validated_image_feature_payload(
                    payload,
                    cache_identity=self.cache_identity,
                    model_name=self.model_name,
                    precision=self.precision,
                    dtype_name=self._dtype_name,
                    cache_tag=cache_tag,
                    sample_count=sample_count,
                )
                if cached is not None:
                    return cached
            except (
                EOFError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                pickle.UnpicklingError,
            ):
                pass

        selected_dataset: Dataset[Any]
        if sample_count == dataset_size:
            selected_dataset = dataset
        else:
            selected_dataset = Subset(dataset, range(sample_count))
        def encode_rows(start: int, stop: int) -> ImageFeatureSet:
            row_dataset: Dataset[Any]
            if start == 0 and stop == sample_count:
                row_dataset = selected_dataset
            else:
                row_dataset = Subset(selected_dataset, range(start, stop))
            generator = torch.Generator().manual_seed(seed + start)
            loader = DataLoader(
                row_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=self.device.type == "cuda",
                generator=generator,
            )
            feature_batches: list[Tensor] = []
            label_batches: list[Tensor] = []
            processed = 0
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    raise ValueError("Dataset items must contain (image, target)")
                images, labels = batch[0], batch[1]
                features = self.encode_images(images)
                feature_batches.append(features.detach().cpu())
                label_batches.append(torch.as_tensor(labels, dtype=torch.long).cpu())
                processed += features.shape[0]
                if progress is not None:
                    progress("image", start + processed, sample_count)
            if processed != stop - start:
                raise RuntimeError(
                    f"Image loader returned {processed} rows for [{start}, {stop})"
                )
            return ImageFeatureSet(
                features=torch.cat(feature_batches, dim=0).contiguous(),
                labels=torch.cat(label_batches, dim=0).long().contiguous(),
                sample_indices=torch.arange(start, stop, dtype=torch.long),
            )

        if cache_shard_size is None:
            result = encode_rows(0, sample_count)
        else:
            assert resolved_cache is not None
            shard_directory = resolved_cache.with_name(f"{resolved_cache.name}.shards")
            shard_directory.mkdir(parents=True, exist_ok=True)
            shards: list[ImageFeatureSet] = []
            for start in range(0, sample_count, cache_shard_size):
                stop = min(start + cache_shard_size, sample_count)
                shard_path = shard_directory / f"part-{start:09d}-{stop:09d}.pt"
                shard: ImageFeatureSet | None = None
                if shard_path.is_file():
                    try:
                        shard = _validated_image_shard_payload(
                            torch_load_tensors(shard_path),
                            cache_identity=self.cache_identity,
                            model_name=self.model_name,
                            precision=self.precision,
                            dtype_name=self._dtype_name,
                            cache_tag=cache_tag,
                            sample_count=sample_count,
                            start=start,
                            stop=stop,
                        )
                    except (
                        EOFError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        pickle.UnpicklingError,
                    ):
                        shard = None
                if shard is None:
                    shard = encode_rows(start, stop)
                    atomic_torch_save(
                        {
                            "format": _IMAGE_SHARD_FORMAT,
                            "cache_identity": self.cache_identity,
                            "model_name": self.model_name,
                            "precision": self.precision,
                            "dtype": self._dtype_name,
                            "cache_tag": cache_tag,
                            "sample_count": sample_count,
                            "shard_start": start,
                            "shard_stop": stop,
                            "features": shard.features,
                            "labels": shard.labels,
                            "sample_indices": shard.sample_indices,
                        },
                        shard_path,
                    )
                elif progress is not None:
                    progress("image-cache", stop, sample_count)
                shards.append(shard)
            result = ImageFeatureSet(
                features=torch.cat([shard.features for shard in shards], dim=0).contiguous(),
                labels=torch.cat([shard.labels for shard in shards], dim=0)
                .long()
                .contiguous(),
                sample_indices=torch.cat(
                    [shard.sample_indices for shard in shards], dim=0
                )
                .long()
                .contiguous(),
            )
        if resolved_cache is not None:
            atomic_torch_save(
                {
                    "format": _IMAGE_CACHE_FORMAT,
                    "cache_identity": self.cache_identity,
                    "model_name": self.model_name,
                    "precision": self.precision,
                    "dtype": self._dtype_name,
                    "cache_tag": cache_tag,
                    "sample_count": sample_count,
                    "features": result.features,
                    "labels": result.labels,
                    "sample_indices": result.sample_indices,
                },
                resolved_cache,
            )
        return result
