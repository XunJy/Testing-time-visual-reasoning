"""Vision-language model backends used by experiment combinations."""

from .base import (
    ImageFeatureSet,
    ProgressCallback,
    TextFeatureTable,
    VisionLanguageBackend,
)
from .clip import DEFAULT_CLIP_MODEL, CLIPBackend, ClipPrecision
from .open_clip import (
    EVA02_CLIP_L14_336,
    OPEN_CLIP_TORCH_VERSION,
    TIMM_VERSION,
    OpenCLIPBackend,
    OpenCLIPCheckpoint,
    OpenCLIPPrecision,
)

__all__ = [
    "CLIPBackend",
    "DEFAULT_CLIP_MODEL",
    "ClipPrecision",
    "ImageFeatureSet",
    "EVA02_CLIP_L14_336",
    "OPEN_CLIP_TORCH_VERSION",
    "OpenCLIPBackend",
    "OpenCLIPCheckpoint",
    "OpenCLIPPrecision",
    "ProgressCallback",
    "TIMM_VERSION",
    "TextFeatureTable",
    "VisionLanguageBackend",
]
