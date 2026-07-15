"""Common feature containers and interface for vision-language model backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor
from torch.utils.data import Dataset

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class ImageFeatureSet:
    """Cached image features and aligned labels/sample positions."""

    features: Tensor
    labels: Tensor
    sample_indices: Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must have shape [samples, dimensions]")
        if self.labels.ndim != 1 or self.sample_indices.ndim != 1:
            raise ValueError("labels and sample_indices must be one-dimensional")
        if not (self.features.shape[0] == self.labels.shape[0] == self.sample_indices.shape[0]):
            raise ValueError("Image features, labels, and indices are misaligned")
        if self.features.shape[0] == 0:
            raise ValueError("ImageFeatureSet must not be empty")

    @property
    def size(self) -> int:
        return self.features.shape[0]


@dataclass(frozen=True, slots=True)
class TextFeatureTable:
    """A dense device-resident table for prompt aggregation."""

    texts: tuple[str, ...]
    features: Tensor
    index: dict[str, int]

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[0] != len(self.texts):
            raise ValueError("Text table rows must align with texts")
        if len(self.index) != len(self.texts):
            raise ValueError("Text table index must contain every unique text")


class VisionLanguageBackend(Protocol):
    """Structural interface required by model-agnostic TT-VR methods."""

    model_name: str
    cache_identity: str
    device: torch.device
    precision: str
    text_batch_size: int
    preprocess: Callable[[Any], Tensor]

    @property
    def feature_dtype_name(self) -> str: ...

    def configure_text_cache(self, path: Path | str) -> None: ...

    def save_text_cache(self) -> Path | None: ...

    def build_text_feature_table(
        self,
        texts: Iterable[str],
        *,
        progress: ProgressCallback | None = None,
    ) -> TextFeatureTable: ...

    def pool_prompt_groups(
        self,
        groups: Sequence[Sequence[str]],
        table: TextFeatureTable | None = None,
    ) -> Tensor: ...

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
    ) -> ImageFeatureSet: ...
