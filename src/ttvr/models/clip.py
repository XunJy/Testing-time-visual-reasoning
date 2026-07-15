"""Reusable OpenAI CLIP inference and feature-cache backend."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from .base import ImageFeatureSet, ProgressCallback, TextFeatureTable

DEFAULT_CLIP_MODEL = "ViT-L/14@336px"
ClipPrecision = Literal["fp32", "native"]

_TEXT_CACHE_FORMAT = 1
_IMAGE_CACHE_FORMAT = 1


def _torch_load(path: Path) -> Any:
    """Load tensor-only caches safely across supported PyTorch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 does not expose weights_only.
        return torch.load(path, map_location="cpu")


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalise(features: Tensor) -> Tensor:
    return functional.normalize(features, dim=-1)


class CLIPBackend:
    """OpenAI CLIP encoder with batched in-memory and on-disk caches.

    The default model is the paper's large CUB backbone.  Importing this module
    does not import ``clip``; the dependency is loaded only when constructing a
    backend, keeping prompt and metric utilities lightweight.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        *,
        device: str | torch.device | None = None,
        precision: ClipPrecision = "fp32",
        text_batch_size: int = 256,
        model_cache_dir: Path | str | None = None,
        text_cache_path: Path | str | None = None,
    ) -> None:
        if text_batch_size <= 0:
            raise ValueError("text_batch_size must be positive")
        if precision not in ("fp32", "native"):
            raise ValueError("precision must be 'fp32' or 'native'")
        try:
            import clip  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "OpenAI CLIP is not installed. Install the official "
                "openai/CLIP package before constructing CLIPBackend."
            ) from error

        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.precision = precision
        self.text_batch_size = text_batch_size
        self._clip = clip
        download_root = None if model_cache_dir is None else str(Path(model_cache_dir).expanduser())
        # The published FuDD implementation loads CLIP on CPU and only then
        # moves it to CUDA.  OpenAI CLIP converts CPU-loaded weights to FP32,
        # so reproducing that order is scientifically meaningful.  ``native``
        # is retained only for explicitly named follow-up experiments.
        load_device: str | torch.device = "cpu" if precision == "fp32" else self.device
        self.model, self.preprocess = clip.load(
            model_name,
            device=load_device,
            jit=False,
            download_root=download_root,
        )
        if precision == "fp32" and self.device.type != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        self._dtype_name = str(next(self.model.parameters()).dtype)

        # Each block is contiguous; locations avoid 150k+ independent tensor
        # storages when the complete CUB prompt cache is serialised.
        self._text_blocks: list[Tensor] = []
        self._text_locations: dict[str, tuple[int, int]] = {}
        self._text_cache_path: Path | None = None
        if text_cache_path is not None:
            self.configure_text_cache(text_cache_path)

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
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
        """Model feature dtype, used to isolate numerically distinct caches."""

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

        payload = _torch_load(cache_path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid text cache payload: {cache_path}")
        metadata_matches = (
            payload.get("format") == _TEXT_CACHE_FORMAT
            and payload.get("model_name") == self.model_name
            and payload.get("precision") == self.precision
            and payload.get("dtype") == self._dtype_name
        )
        if not metadata_matches:
            return
        blocks = payload.get("blocks")
        locations = payload.get("locations")
        if not isinstance(blocks, list) or not all(
            isinstance(block, Tensor) and block.ndim == 2 for block in blocks
        ):
            raise RuntimeError(f"Invalid tensor blocks in text cache: {cache_path}")
        if not isinstance(locations, dict):
            raise RuntimeError(f"Invalid locations in text cache: {cache_path}")

        parsed_locations: dict[str, tuple[int, int]] = {}
        for text, location in locations.items():
            if not (
                isinstance(text, str)
                and isinstance(location, (list, tuple))
                and len(location) == 2
                and all(isinstance(value, int) for value in location)
            ):
                raise RuntimeError(f"Invalid location entry in {cache_path}")
            block_index, row_index = location
            if not (
                0 <= block_index < len(blocks) and 0 <= row_index < blocks[block_index].shape[0]
            ):
                raise RuntimeError(f"Out-of-range location in {cache_path}")
            parsed_locations[text] = (block_index, row_index)
        self._text_blocks = [block.contiguous() for block in blocks]
        self._text_locations = parsed_locations

    def save_text_cache(self) -> Path | None:
        """Atomically persist all cached text features, if a path is configured."""

        if self._text_cache_path is None:
            return None
        payload = {
            "format": _TEXT_CACHE_FORMAT,
            "model_name": self.model_name,
            "precision": self.precision,
            "dtype": self._dtype_name,
            "blocks": self._text_blocks,
            "locations": self._text_locations,
        }
        _atomic_torch_save(payload, self._text_cache_path)
        return self._text_cache_path

    @staticmethod
    def _validated_texts(texts: Iterable[str]) -> tuple[str, ...]:
        result = tuple(texts)
        if not result:
            raise ValueError("At least one text is required")
        if any(not isinstance(text, str) or not text.strip() for text in result):
            raise ValueError("All prompts must be non-empty strings")
        return result

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
                tokens = self._clip.tokenize(list(chunk)).to(self.device)
                features = _normalise(self.model.encode_text(tokens))
                block = features.detach().cpu().contiguous()
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
        """Mean-pool normalised prompt features, then normalise each group.

        This is the ``similarity of mean embeddings`` aggregation used by the
        official FuDD implementation.
        """

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

        indices = torch.tensor(
            flat_indices,
            dtype=torch.long,
            device=table.features.device,
        )
        selected = table.features.index_select(0, indices)
        length_tensor = torch.tensor(
            lengths,
            dtype=torch.long,
            device=table.features.device,
        )
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
        return _normalise(means)

    def encode_images(self, images: Tensor) -> Tensor:
        """Encode a preprocessed image batch and L2-normalise each feature."""

        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError(
                "images must be a [batch, channels, height, width] tensor; "
                "pass CLIPBackend.preprocess to the dataset"
            )
        with torch.inference_mode():
            return _normalise(self.model.encode_image(images.to(self.device, non_blocking=True)))

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
        progress: ProgressCallback | None = None,
    ) -> ImageFeatureSet:
        """Batch-encode a deterministic prefix of a dataset, with disk caching."""

        if batch_size <= 0 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        dataset_size = len(dataset)
        if dataset_size <= 0:
            raise ValueError("dataset must not be empty")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        sample_count = min(dataset_size, max_samples or dataset_size)
        resolved_cache = None if cache_path is None else Path(cache_path).expanduser()

        if resolved_cache is not None and resolved_cache.is_file():
            try:
                payload = _torch_load(resolved_cache)
                metadata_matches = (
                    isinstance(payload, dict)
                    and payload.get("format") == _IMAGE_CACHE_FORMAT
                    and payload.get("model_name") == self.model_name
                    and payload.get("precision") == self.precision
                    and payload.get("dtype") == self._dtype_name
                    and payload.get("cache_tag") == cache_tag
                    and payload.get("sample_count") == sample_count
                )
                if metadata_matches:
                    return ImageFeatureSet(
                        features=payload["features"],
                        labels=payload["labels"].long(),
                        sample_indices=payload["sample_indices"].long(),
                    )
            except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError):
                pass

        selected_dataset: Dataset[Any]
        if sample_count == dataset_size:
            selected_dataset = dataset
        else:
            selected_dataset = Subset(dataset, range(sample_count))
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            selected_dataset,
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
                progress("image", processed, sample_count)

        result = ImageFeatureSet(
            features=torch.cat(feature_batches, dim=0).contiguous(),
            labels=torch.cat(label_batches, dim=0).long().contiguous(),
            sample_indices=torch.arange(sample_count, dtype=torch.long),
        )
        if resolved_cache is not None:
            _atomic_torch_save(
                {
                    "format": _IMAGE_CACHE_FORMAT,
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
