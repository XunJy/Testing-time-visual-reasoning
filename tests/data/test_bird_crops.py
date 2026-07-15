from __future__ import annotations

import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from ttvr.data.bird_crops import (
    SQUARE_CONTEXT_RECIPE_ID,
    clip_bbox_xyxy_to_image,
    coco_bbox_to_xyxy,
    md5_base64_file,
    md5_file,
    save_crop_png_atomic,
    square_context_crop,
    validate_relative_posix_path,
)


def test_square_context_crop_is_deterministic_square_and_records_padding() -> None:
    image = Image.new("RGB", (10, 8), color=(20, 30, 40))

    first, geometry = square_context_crop(image, (0.0, 1.0, 4.0, 3.0))
    second, repeated = square_context_crop(image, (0.0, 1.0, 4.0, 3.0))

    assert first.size == second.size == (5, 5)
    assert first.tobytes() == second.tobytes()
    assert geometry == repeated
    assert geometry.recipe_id == SQUARE_CONTEXT_RECIPE_ID
    assert geometry.crop_box_xyxy == (-1, -1, 4, 4)
    assert geometry.padding_ltrb == (1, 1, 0, 0)
    assert first.getpixel((0, 0)) == (0, 0, 0)


def test_crop_contains_fractional_integer_envelope() -> None:
    image = Image.new("RGB", (20, 20))
    _crop, geometry = square_context_crop(
        image,
        (1.2, 3.1, 10.9, 8.2),
        context_scale=1.0,
    )
    left, top, right, bottom = geometry.crop_box_xyxy

    assert left <= 1
    assert top <= 3
    assert right >= 11
    assert bottom >= 9
    assert right - left == bottom - top


def test_clipping_is_explicit_and_disjoint_boxes_fail_closed() -> None:
    clipped, changed = clip_bbox_xyxy_to_image((-0.5, 2.0, 4.0, 7.5), (6, 6))

    assert clipped == (0.0, 2.0, 4.0, 6.0)
    assert changed
    with pytest.raises(ValueError, match="clip it explicitly"):
        square_context_crop(Image.new("RGB", (6, 6)), (-0.5, 2.0, 4.0, 5.0))
    with pytest.raises(ValueError, match="does not overlap"):
        clip_bbox_xyxy_to_image((7.0, 7.0, 8.0, 8.0), (6, 6))


@pytest.mark.parametrize(
    "bbox",
    [
        (0, 0, 0, 1),
        (0, 0, 1, -1),
        (0, 0, float("nan"), 1),
        (0, 0, 1),
    ],
)
def test_invalid_boxes_fail_closed(bbox: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        square_context_crop(Image.new("RGB", (4, 4)), bbox)


def test_coco_conversion_and_context_scale_validation() -> None:
    assert coco_bbox_to_xyxy((2, 3, 4, 5)) == (2.0, 3.0, 6.0, 8.0)
    with pytest.raises(ValueError, match="positive"):
        coco_bbox_to_xyxy((2, 3, -1, 5))
    with pytest.raises(ValueError, match="at least"):
        square_context_crop(Image.new("RGB", (10, 10)), (1, 1, 2, 2), context_scale=0.5)


@pytest.mark.parametrize(
    "value",
    ["../bird.jpg", "/bird.jpg", "folder/./bird.jpg", "folder\\bird.jpg", "bird\n.jpg"],
)
def test_unsafe_relative_paths_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="relative path"):
        validate_relative_posix_path(value)
    assert validate_relative_posix_path("flight/a-b_1.jpg") == "flight/a-b_1.jpg"


def test_md5_helpers_and_atomic_crop_writer(tmp_path: Path) -> None:
    payload = b"publisher object"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    expected_hex = hashlib.md5(payload).hexdigest()  # noqa: S324 - fixture checksum.
    expected_base64 = base64.b64encode(bytes.fromhex(expected_hex)).decode("ascii")

    assert md5_file(source) == expected_hex
    assert md5_base64_file(source) == expected_base64

    destination = tmp_path / "crop.png"
    digest, phash = save_crop_png_atomic(Image.new("RGB", (3, 3), "red"), destination)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
    assert len(phash) == 16
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_crop_png_atomic(Image.new("RGB", (3, 3), "blue"), destination)


def test_crop_writer_has_atomic_no_replace_semantics(tmp_path: Path) -> None:
    destination = tmp_path / "one-winner.png"

    def write(colour: str) -> tuple[str, str] | type[Exception]:
        try:
            return save_crop_png_atomic(Image.new("RGB", (8, 8), colour), destination)
        except Exception as error:  # Captured to inspect both concurrent outcomes.
            return type(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write, ("red", "blue")))

    assert sum(outcome is FileExistsError for outcome in outcomes) == 1
    assert sum(isinstance(outcome, tuple) for outcome in outcomes) == 1
    with Image.open(destination) as image:
        assert image.getpixel((0, 0)) in {(255, 0, 0), (0, 0, 255)}
