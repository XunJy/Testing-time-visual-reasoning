"""Audited, deterministic bounding-box crops shared by bird data sources.

The versioned recipe keeps a small amount of visual context while making every
derived image square:

1. validate a finite ``(x_min, y_min, x_max, y_max)`` box;
2. choose ``ceil(context_scale * max(box_width, box_height))`` pixels, enlarged
   if necessary to contain the integer envelope of the box;
3. centre that square on the box (ties are resolved with ``floor``); and
4. let Pillow fill any area outside the source image with zero-valued pixels.

Callers that consume publisher boxes extending beyond an image must explicitly
call :func:`clip_bbox_xyxy_to_image` first and preserve both the original and
effective boxes in their provenance.  The crop function itself fails closed on
out-of-bounds boxes.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from .bird_manifest import perceptual_dhash

SQUARE_CONTEXT_RECIPE_ID = "ttvr-square-context-v1"
DEFAULT_CONTEXT_SCALE = 1.25


@dataclass(frozen=True, slots=True)
class SquareCropGeometry:
    """Complete geometry needed to reproduce one square context crop."""

    recipe_id: str
    source_size: tuple[int, int]
    bbox_xyxy: tuple[float, float, float, float]
    context_scale: float
    crop_box_xyxy: tuple[int, int, int, int]
    padding_ltrb: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def md5_file(path: Path | str) -> str:
    """Return a file MD5 in lowercase hexadecimal for publisher verification."""

    digest = hashlib.md5()  # noqa: S324 - verifies publisher-provided checksums.
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_base64_file(path: Path | str) -> str:
    """Return the RFC 4648 base64 form used by Google Cloud Storage ``md5Hash``."""

    raw = bytes.fromhex(md5_file(path))
    return base64.b64encode(raw).decode("ascii")


def validate_relative_posix_path(value: str) -> str:
    """Validate an untrusted publisher path without changing its identity."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("relative path must be a non-empty, surrounding-space-free string")
    if "\\" in value or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"unsafe relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    if str(path) != value:
        raise ValueError(f"relative path is not canonical POSIX form: {value!r}")
    return value


def _finite_box(bbox_xyxy: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    try:
        box = tuple(float(value) for value in bbox_xyxy)
    except (TypeError, ValueError) as error:
        raise ValueError("bbox coordinates must be numeric") from error
    if not all(math.isfinite(value) for value in box):
        raise ValueError("bbox coordinates must be finite")
    x_min, y_min, x_max, y_max = box
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bbox must have positive width and height")
    return x_min, y_min, x_max, y_max


def coco_bbox_to_xyxy(
    bbox_xywh: Sequence[float],
) -> tuple[float, float, float, float]:
    """Convert and validate a COCO ``(x, y, width, height)`` box."""

    if len(bbox_xywh) != 4:
        raise ValueError("COCO bbox must contain exactly four coordinates")
    try:
        x_min, y_min, width, height = (float(value) for value in bbox_xywh)
    except (TypeError, ValueError) as error:
        raise ValueError("COCO bbox coordinates must be numeric") from error
    if not all(math.isfinite(value) for value in (x_min, y_min, width, height)):
        raise ValueError("COCO bbox coordinates must be finite")
    if width <= 0 or height <= 0:
        raise ValueError("COCO bbox must have positive width and height")
    return _finite_box((x_min, y_min, x_min + width, y_min + height))


def clip_bbox_xyxy_to_image(
    bbox_xyxy: Sequence[float],
    image_size: tuple[int, int],
) -> tuple[tuple[float, float, float, float], bool]:
    """Clip a partially visible box and report whether publisher geometry changed.

    Completely disjoint boxes still fail closed.  This explicit operation is
    separate from :func:`square_context_crop` so callers cannot silently hide
    publisher coordinate defects.
    """

    box = _finite_box(bbox_xyxy)
    width, height = image_size
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("image_size must contain two positive integers")
    x_min, y_min, x_max, y_max = box
    clipped = (
        max(0.0, min(float(width), x_min)),
        max(0.0, min(float(height), y_min)),
        max(0.0, min(float(width), x_max)),
        max(0.0, min(float(height), y_max)),
    )
    try:
        clipped = _finite_box(clipped)
    except ValueError as error:
        raise ValueError("bbox does not overlap the source image") from error
    return clipped, clipped != box


def square_context_crop(
    image: Image.Image,
    bbox_xyxy: Sequence[float],
    *,
    context_scale: float = DEFAULT_CONTEXT_SCALE,
) -> tuple[Image.Image, SquareCropGeometry]:
    """Return a deterministic square crop and its exact integer geometry."""

    if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError("image must be a non-empty PIL image")
    if not isinstance(context_scale, (int, float)) or not math.isfinite(context_scale):
        raise ValueError("context_scale must be finite")
    context_scale = float(context_scale)
    if context_scale < 1.0:
        raise ValueError("context_scale must be at least 1.0")
    x_min, y_min, x_max, y_max = _finite_box(bbox_xyxy)
    if x_min < 0 or y_min < 0 or x_max > image.width or y_max > image.height:
        raise ValueError("bbox lies outside the source image; clip it explicitly first")

    box_width = x_max - x_min
    box_height = y_max - y_min
    integer_width = math.ceil(x_max) - math.floor(x_min)
    integer_height = math.ceil(y_max) - math.floor(y_min)
    side = max(
        1,
        math.ceil(context_scale * max(box_width, box_height)),
        integer_width,
        integer_height,
    )
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    def _origin(center: float, lower: float, upper: float) -> int:
        candidate = math.floor(center - side / 2.0)
        lowest = math.ceil(upper) - side
        highest = math.floor(lower)
        return min(highest, max(lowest, candidate))

    left = _origin(center_x, x_min, x_max)
    top = _origin(center_y, y_min, y_max)
    right = left + side
    bottom = top + side
    padding = (
        max(0, -left),
        max(0, -top),
        max(0, right - image.width),
        max(0, bottom - image.height),
    )
    crop = image.crop((left, top, right, bottom))
    if crop.size != (side, side):  # pragma: no cover - defensive Pillow invariant.
        raise RuntimeError("Pillow returned a non-square crop")
    geometry = SquareCropGeometry(
        recipe_id=SQUARE_CONTEXT_RECIPE_ID,
        source_size=(image.width, image.height),
        bbox_xyxy=(x_min, y_min, x_max, y_max),
        context_scale=context_scale,
        crop_box_xyxy=(left, top, right, bottom),
        padding_ltrb=padding,
    )
    return crop, geometry


def _encode_crop_png(image: Image.Image) -> bytes:
    """Encode a crop losslessly with fixed Pillow parameters."""

    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=6, optimize=False)
    return output.getvalue()


def save_crop_png_atomic(
    image: Image.Image,
    destination: Path | str,
) -> tuple[str, str]:
    """Write a new lossless crop atomically and return SHA-256 plus dHash.

    Existing paths are never replaced.  This mirrors the immutable manifest
    writers and keeps interrupted preparation runs auditable.
    """

    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite crop: {path}")
    payload = _encode_crop_png(image)
    digest = hashlib.sha256(payload).hexdigest()
    phash = perceptual_dhash(image)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.part-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A hard link is the portable same-filesystem no-replace primitive:
            # only one concurrent preparer can create ``path``.  ``replace``
            # would have a TOCTOU window and could overwrite another crop.
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite crop: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return digest, phash
