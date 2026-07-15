"""OpenAI CLIP backend used by the paper-faithful FuDD reproduction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from .cached import CachedFeatureBackend, normalise_features

DEFAULT_CLIP_MODEL = "ViT-L/14@336px"
OPENAI_CLIP_COMMIT = "a1d071733d7111c9c014f024669f959182114e33"
ClipPrecision = Literal["fp32", "native"]


class CLIPBackend(CachedFeatureBackend):
    """OpenAI CLIP encoder with deterministic feature caches.

    The default model and FP32 loading order reproduce the FuDD paper's CUB
    backbone.  Importing this module does not import ``clip``; the dependency is
    loaded only when constructing a backend.
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

        resolved_device = self.resolve_device(device)
        download_root = None if model_cache_dir is None else str(Path(model_cache_dir).expanduser())
        # The published FuDD implementation loads CLIP on CPU and only then
        # moves it to CUDA. OpenAI CLIP converts CPU-loaded weights to FP32, so
        # reproducing that order is scientifically meaningful.
        load_device: str | torch.device = "cpu" if precision == "fp32" else resolved_device
        self.model, self.preprocess = clip.load(
            model_name,
            device=load_device,
            jit=False,
            download_root=download_root,
        )
        if precision == "fp32" and resolved_device.type != "cpu":
            self.model = self.model.to(resolved_device)
        self.model.eval()
        self._clip = clip
        feature_dtype_name = str(next(self.model.parameters()).dtype)
        super().__init__(
            model_name=model_name,
            cache_identity=f"openai-clip:{model_name}@{OPENAI_CLIP_COMMIT}",
            device=resolved_device,
            precision=precision,
            text_batch_size=text_batch_size,
            feature_dtype_name=feature_dtype_name,
            text_cache_path=text_cache_path,
        )

    def _encode_text_batch(self, texts: Sequence[str]) -> Tensor:
        tokens = self._clip.tokenize(list(texts)).to(self.device)
        return normalise_features(self.model.encode_text(tokens))

    def encode_images(self, images: Tensor) -> Tensor:
        """Encode a preprocessed image batch and L2-normalise each feature."""

        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError(
                "images must be a [batch, channels, height, width] tensor; "
                "pass CLIPBackend.preprocess to the dataset"
            )
        with torch.inference_mode():
            return normalise_features(
                self.model.encode_image(images.to(self.device, non_blocking=True))
            )
