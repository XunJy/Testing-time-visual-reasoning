from __future__ import annotations

from pathlib import Path

import pytest

from ttvr.data.cub import (
    CUB_CLASS_COUNT,
    CUB_DIRECTORY_NAME,
    CUB_IMAGE_COUNT,
    CUB_TEST_COUNT,
    CUB_TRAIN_COUNT,
    CUB200Dataset,
    canonical_class_name,
    validate_class_name_alignment,
    validate_cub,
)


@pytest.fixture
def synthetic_cub_root(tmp_path: Path) -> Path:
    """Write official-sized metadata without copying any CUB image content."""

    dataset_root = tmp_path / CUB_DIRECTORY_NAME
    dataset_root.mkdir()
    image_rows: list[str] = []
    label_rows: list[str] = []
    split_rows: list[str] = []
    class_rows = [
        f"{class_id} {class_id:03d}.Synthetic_bird_{class_id:03d}"
        for class_id in range(1, CUB_CLASS_COUNT + 1)
    ]
    for image_id in range(1, CUB_IMAGE_COUNT + 1):
        class_id = (image_id - 1) % CUB_CLASS_COUNT + 1
        image_rows.append(f"{image_id} {class_id:03d}/image_{image_id:05d}.jpg")
        label_rows.append(f"{image_id} {class_id}")
        is_training = int(image_id <= CUB_TRAIN_COUNT)
        split_rows.append(f"{image_id} {is_training}")

    (dataset_root / "images.txt").write_text(
        "\n".join(image_rows) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "classes.txt").write_text(
        "\n".join(class_rows) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "image_class_labels.txt").write_text(
        "\n".join(label_rows) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "train_test_split.txt").write_text(
        "\n".join(split_rows) + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_validate_cub_enforces_official_counts(synthetic_cub_root: Path) -> None:
    report = validate_cub(synthetic_cub_root, verify_images=False)

    assert report.image_count == CUB_IMAGE_COUNT
    assert report.train_count == CUB_TRAIN_COUNT
    assert report.test_count == CUB_TEST_COUNT
    assert report.class_count == CUB_CLASS_COUNT
    assert not report.checked_image_files


def test_test_dataset_uses_official_split_and_zero_based_targets(
    synthetic_cub_root: Path,
) -> None:
    dataset = CUB200Dataset(
        synthetic_cub_root,
        split="test",
        verify_images=False,
    )

    assert len(dataset) == CUB_TEST_COUNT
    assert dataset.samples[0].image_id == CUB_TRAIN_COUNT + 1
    assert min(dataset.targets) == 0
    assert max(dataset.targets) == CUB_CLASS_COUNT - 1
    assert dataset.class_names[:2] == (
        "Synthetic bird 001",
        "Synthetic bird 002",
    )
    fingerprint = dataset.fingerprint
    same_split = CUB200Dataset(
        synthetic_cub_root,
        split="test",
        verify_images=False,
    )
    assert len(fingerprint) == 64
    assert same_split.fingerprint == fingerprint


def test_dataset_rejects_unknown_split(synthetic_cub_root: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        CUB200Dataset(
            synthetic_cub_root,
            split="validation",  # type: ignore[arg-type]
            verify_images=False,
        )


def test_class_name_alignment_ignores_only_formatting_differences() -> None:
    cub_names = [f"Bird class {index}" for index in range(CUB_CLASS_COUNT)]
    prompt_names = [f"bird-class_{index}" for index in range(CUB_CLASS_COUNT)]

    validate_class_name_alignment(cub_names, prompt_names)
    assert canonical_class_name("Black footed_Albatross") == canonical_class_name(
        "Black-footed Albatross"
    )


def test_class_name_alignment_rejects_reordered_labels() -> None:
    cub_names = [f"Bird {index}" for index in range(CUB_CLASS_COUNT)]
    prompt_names = cub_names.copy()
    prompt_names[40], prompt_names[41] = prompt_names[41], prompt_names[40]

    with pytest.raises(ValueError, match="zero-based id 40"):
        validate_class_name_alignment(cub_names, prompt_names)
