"""OpenCLIP backends with immutable checkpoint provenance."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from .cached import CachedFeatureBackend, normalise_features

OpenCLIPPrecision = Literal["fp16", "fp32"]

OPEN_CLIP_TORCH_VERSION = "3.3.0"
TIMM_VERSION = "1.0.28"


@dataclass(frozen=True, slots=True)
class OpenCLIPCheckpoint:
    """Complete model, tokenizer, checkpoint, and preprocessing identity."""

    model_name: str
    pretrained_tag: str
    hf_repo_id: str
    hf_revision: str
    checkpoint_filename: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    tokenizer: str
    context_length: int
    image_size: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    interpolation: str
    resize_mode: str
    crop_mode: str

    def __post_init__(self) -> None:
        text_values = (
            self.model_name,
            self.pretrained_tag,
            self.hf_repo_id,
            self.hf_revision,
            self.checkpoint_filename,
            self.tokenizer,
            self.interpolation,
            self.resize_mode,
            self.crop_mode,
        )
        if any(not value.strip() for value in text_values):
            raise ValueError("Checkpoint string fields must not be empty")
        if len(self.hf_revision) != 40 or any(
            c not in "0123456789abcdef" for c in self.hf_revision
        ):
            raise ValueError("hf_revision must be a 40-character lowercase commit hash")
        if len(self.checkpoint_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.checkpoint_sha256
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        if self.checkpoint_bytes <= 0 or self.context_length <= 0 or self.image_size <= 0:
            raise ValueError("Checkpoint size, context length, and image size must be positive")

    @property
    def cache_identity(self) -> str:
        """Identity that prevents caches crossing checkpoints or tokenizers."""

        values = (
            "open-clip",
            self.model_name,
            self.pretrained_tag,
            self.hf_repo_id,
            self.hf_revision,
            self.checkpoint_filename,
            self.checkpoint_sha256,
            self.tokenizer,
            str(self.context_length),
            str(self.image_size),
            repr(self.image_mean),
            repr(self.image_std),
            self.interpolation,
            self.resize_mode,
            self.crop_mode,
        )
        return "|".join(values)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe checkpoint provenance."""

        values = asdict(self)
        values["image_mean"] = list(self.image_mean)
        values["image_std"] = list(self.image_std)
        return values


EVA02_CLIP_L14_336 = OpenCLIPCheckpoint(
    model_name="EVA02-L-14-336",
    pretrained_tag="merged2b_s6b_b61k",
    hf_repo_id="timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k",
    hf_revision="4f62907359c8506be7021582f360564693b22c15",
    checkpoint_filename="open_clip_model.safetensors",
    checkpoint_sha256="f753bca0e8327f77e8845b0af2510d599c3e4614237007b48078c791f2cf391c",
    checkpoint_bytes=856_239_456,
    tokenizer="open_clip.SimpleTokenizer/openai-bpe",
    context_length=77,
    image_size=336,
    image_mean=(0.48145466, 0.4578275, 0.40821073),
    image_std=(0.26862954, 0.26130258, 0.27577711),
    interpolation="bicubic",
    resize_mode="shortest",
    crop_mode="center",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_hf_checkpoint(
    checkpoint: OpenCLIPCheckpoint,
    cache_dir: Path | str | None,
) -> Path:
    """Download an immutable safetensors revision and verify bytes and SHA-256."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface-hub is required for OpenCLIP checkpoints; install the eva02 extra"
        ) from error

    resolved_cache = None if cache_dir is None else str(Path(cache_dir).expanduser())
    path = Path(
        hf_hub_download(
            repo_id=checkpoint.hf_repo_id,
            filename=checkpoint.checkpoint_filename,
            revision=checkpoint.hf_revision,
            cache_dir=resolved_cache,
        )
    )
    actual_size = path.stat().st_size
    if actual_size != checkpoint.checkpoint_bytes:
        raise RuntimeError(
            f"Checkpoint byte size mismatch for {path}: "
            f"expected {checkpoint.checkpoint_bytes}, got {actual_size}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != checkpoint.checkpoint_sha256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch for {path}: "
            f"expected {checkpoint.checkpoint_sha256}, got {actual_sha256}"
        )
    return path


class OpenCLIPBackend(CachedFeatureBackend):
    """OpenCLIP dual encoder with verified weights and FP32 feature math.

    The model forward may use FP16 on a T4, but image/text embeddings are cast
    to FP32 before L2 normalisation, caching, prompt means, and similarities.
    """

    def __init__(
        self,
        checkpoint: OpenCLIPCheckpoint = EVA02_CLIP_L14_336,
        *,
        device: str | torch.device | None = None,
        precision: OpenCLIPPrecision = "fp16",
        text_batch_size: int = 256,
        model_cache_dir: Path | str | None = None,
        text_cache_path: Path | str | None = None,
    ) -> None:
        if text_batch_size <= 0:
            raise ValueError("text_batch_size must be positive")
        if precision not in ("fp16", "fp32"):
            raise ValueError("precision must be 'fp16' or 'fp32'")
        resolved_device = self.resolve_device(device)
        if precision == "fp16" and resolved_device.type != "cuda":
            raise ValueError("fp16 OpenCLIP inference requires a CUDA device")
        try:
            import open_clip  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "OpenCLIP is not installed. Install this project with the eva02 extra."
            ) from error

        checkpoint_path = _verified_hf_checkpoint(checkpoint, model_cache_dir)
        cache_dir = None if model_cache_dir is None else str(Path(model_cache_dir).expanduser())
        model, _, preprocess = open_clip.create_model_and_transforms(
            checkpoint.model_name,
            pretrained=str(checkpoint_path),
            precision=precision,
            device=resolved_device,
            cache_dir=cache_dir,
            weights_only=True,
            force_image_size=checkpoint.image_size,
            image_mean=checkpoint.image_mean,
            image_std=checkpoint.image_std,
            image_interpolation=checkpoint.interpolation,
            image_resize_mode=checkpoint.resize_mode,
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(
            checkpoint.model_name,
            context_length=checkpoint.context_length,
            cache_dir=cache_dir,
        )

        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.preprocess = preprocess
        self._tokenizer = tokenizer
        self._model_dtype = next(model.parameters()).dtype
        super().__init__(
            model_name=checkpoint.model_name,
            cache_identity=checkpoint.cache_identity,
            device=resolved_device,
            precision=precision,
            text_batch_size=text_batch_size,
            # FP16 is confined to the model forward; all stored features are FP32.
            feature_dtype_name="torch.float32",
            text_cache_path=text_cache_path,
        )

    @property
    def model_dtype_name(self) -> str:
        """Parameter dtype used inside the model forward."""

        return str(self._model_dtype)

    def provenance(self) -> dict[str, Any]:
        """Return the complete identity needed to audit an experiment run."""

        return {
            "checkpoint": self.checkpoint.to_dict(),
            "checkpoint_verification": {
                "status": "verified",
                "actual_bytes": self.checkpoint_path.stat().st_size,
                "actual_sha256": self.checkpoint.checkpoint_sha256,
            },
            "cache_identity_sha256": hashlib.sha256(self.cache_identity.encode()).hexdigest(),
            "model_forward_dtype": self.model_dtype_name,
            "feature_dtype": self.feature_dtype_name,
            "open_clip_torch_required": OPEN_CLIP_TORCH_VERSION,
            "timm_required": TIMM_VERSION,
        }

    def _encode_text_batch(self, texts: Sequence[str]) -> Tensor:
        tokens = self._tokenizer(list(texts))
        if not isinstance(tokens, Tensor):
            raise TypeError("The locked OpenCLIP SimpleTokenizer must return a tensor")
        tokens = tokens.to(self.device, non_blocking=True)
        features = self.model.encode_text(tokens)
        return normalise_features(features.float())

    def encode_images(self, images: Tensor) -> Tensor:
        """Run FP16/FP32 forward, then normalise FP32 image embeddings."""

        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError(
                "images must be a [batch, channels, height, width] tensor; "
                "pass OpenCLIPBackend.preprocess to the dataset"
            )
        with torch.inference_mode():
            images = images.to(
                device=self.device,
                dtype=self._model_dtype,
                non_blocking=True,
            )
            return normalise_features(self.model.encode_image(images).float())
