"""OpenAI CLIP backend used by the paper-faithful FuDD reproduction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from .cached import CachedFeatureBackend, normalise_features

DEFAULT_CLIP_MODEL = "ViT-L/14@336px"
OPENAI_CLIP_COMMIT = "a1d071733d7111c9c014f024669f959182114e33"
OPENAI_CLIP_REPOSITORY = "https://github.com/openai/CLIP.git"
VIT_L14_336_CHECKPOINT_FILENAME = "ViT-L-14-336px.pt"
VIT_L14_336_CHECKPOINT_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
ClipPrecision = Literal["fp32", "native"]


@dataclass(frozen=True, slots=True)
class OpenAIClipInstallation:
    """Audited PEP 610 identity of the installed official CLIP package."""

    distribution: str
    version: str
    repository_url: str
    vcs: str
    commit_id: str


@dataclass(frozen=True, slots=True)
class OpenAIClipCheckpoint:
    """Audited identity of one local OpenAI CLIP weight file."""

    model_name: str
    path: Path
    sha256: str
    size_bytes: int


def _require_lower_hex(value: str, length: int, *, label: str) -> None:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be {length} lowercase hexadecimal characters")


def verify_openai_clip_installation(
    *,
    expected_commit: str,
    expected_repository_url: str = OPENAI_CLIP_REPOSITORY,
) -> OpenAIClipInstallation:
    """Fail closed unless ``clip`` was installed from the expected Git commit."""

    _require_lower_hex(expected_commit, 40, label="expected_commit")
    try:
        distribution = importlib.metadata.distribution("clip")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("OpenAI CLIP distribution metadata is not installed") from error
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        raise RuntimeError("OpenAI CLIP installation has no PEP 610 direct_url.json")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("OpenAI CLIP direct_url.json is invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("OpenAI CLIP direct_url.json must contain an object")
    vcs_info = value.get("vcs_info")
    if not isinstance(vcs_info, dict):
        raise RuntimeError("OpenAI CLIP direct_url.json has no VCS provenance")
    repository_url = value.get("url")
    vcs = vcs_info.get("vcs")
    commit_id = vcs_info.get("commit_id")
    if repository_url != expected_repository_url:
        raise RuntimeError(
            "OpenAI CLIP repository mismatch: "
            f"expected {expected_repository_url}, found {repository_url!r}"
        )
    if vcs != "git":
        raise RuntimeError(f"OpenAI CLIP VCS mismatch: expected 'git', found {vcs!r}")
    if commit_id != expected_commit:
        raise RuntimeError(
            f"OpenAI CLIP commit mismatch: expected {expected_commit}, found {commit_id!r}"
        )
    name = distribution.metadata.get("Name")
    if not isinstance(name, str) or name.lower() != "clip":
        raise RuntimeError(f"OpenAI CLIP distribution name mismatch: {name!r}")
    return OpenAIClipInstallation(
        distribution=name,
        version=distribution.version,
        repository_url=repository_url,
        vcs=vcs,
        commit_id=commit_id,
    )


def verify_openai_clip_checkpoint(
    model_cache_dir: Path | str,
    *,
    model_name: str,
    checkpoint_filename: str,
    expected_sha256: str,
) -> OpenAIClipCheckpoint:
    """Hash and verify a local checkpoint selected by an immutable protocol."""

    _require_lower_hex(expected_sha256, 64, label="expected_sha256")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not checkpoint_filename or Path(checkpoint_filename).name != checkpoint_filename:
        raise ValueError("checkpoint_filename must be one plain filename")
    path = Path(model_cache_dir).expanduser() / checkpoint_filename
    if not path.is_file():
        raise RuntimeError(f"Missing OpenAI CLIP checkpoint: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"OpenAI CLIP checkpoint SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, found {actual}"
        )
    return OpenAIClipCheckpoint(
        model_name=model_name,
        path=path.resolve(),
        sha256=actual,
        size_bytes=path.stat().st_size,
    )


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
        installation = verify_openai_clip_installation(expected_commit=OPENAI_CLIP_COMMIT)
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
        checkpoint = None
        if model_name == DEFAULT_CLIP_MODEL and model_cache_dir is not None:
            checkpoint = verify_openai_clip_checkpoint(
                model_cache_dir,
                model_name=model_name,
                checkpoint_filename=VIT_L14_336_CHECKPOINT_FILENAME,
                expected_sha256=VIT_L14_336_CHECKPOINT_SHA256,
            )
        self.openai_clip_installation = installation
        self.openai_clip_checkpoint = checkpoint
        feature_dtype_name = str(next(self.model.parameters()).dtype)
        super().__init__(
            model_name=model_name,
            cache_identity=f"openai-clip:{model_name}@{installation.commit_id}",
            device=resolved_device,
            precision=precision,
            text_batch_size=text_batch_size,
            feature_dtype_name=feature_dtype_name,
            feature_dim=768,
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
