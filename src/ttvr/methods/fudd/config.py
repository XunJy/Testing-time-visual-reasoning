"""Explicit configuration for the CUB FuDD reproduction.

Keeping every experimental choice in a dataclass makes a run serialisable and
avoids hidden process state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...models.clip import DEFAULT_CLIP_MODEL

OFFICIAL_CLIP_MODEL = DEFAULT_CLIP_MODEL
OFFICIAL_CLIP_PRECISION = "fp32"


@dataclass(frozen=True, slots=True)
class FuDDConfig:
    """Configuration shared by a FuDD method + model experiment.

    Parameters are deliberately limited to choices that affect inference or
    reproducibility.  ``device=None`` selects CUDA when available and CPU
    otherwise.  FuDD reports top-5 accuracy, so ``top_k`` must be at least 5.
    """

    data_root: Path | str
    prompt_root: Path | str
    cache_dir: Path | str | None = None
    model_name: str = OFFICIAL_CLIP_MODEL
    precision: str = OFFICIAL_CLIP_PRECISION
    top_k: int = 10
    batch_size: int = 32
    text_batch_size: int = 256
    num_workers: int = 2
    device: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser())
        object.__setattr__(self, "prompt_root", Path(self.prompt_root).expanduser())
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())

        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not self.precision.strip():
            raise ValueError("precision must not be empty")
        if not 5 <= self.top_k <= 200:
            raise ValueError("top_k must be between 5 and 200")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.text_batch_size <= 0:
            raise ValueError("text_batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def is_official_clip_reproduction(self) -> bool:
        """Whether the paper's FuDD + CLIP settings are selected."""

        return (
            self.model_name == OFFICIAL_CLIP_MODEL
            and self.precision == OFFICIAL_CLIP_PRECISION
            and self.top_k == 10
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the configuration."""

        values = asdict(self)
        for key in ("data_root", "prompt_root", "cache_dir"):
            value = values[key]
            values[key] = None if value is None else str(value)
        return values
