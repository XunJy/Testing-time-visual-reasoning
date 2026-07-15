"""Shared CUB-200-2011 download, integrity checks, and dataset loader.

This module mirrors the split used by the official FuDD code while avoiding a
``pandas`` dependency.  Targets are returned as zero-based indices so that they
align with the official FuDD prompt files and CLIP logits.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from torch.utils.data import Dataset

CUB_ARCHIVE_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
CUB_ARCHIVE_NAME = "CUB_200_2011.tgz"
CUB_ARCHIVE_MD5 = "97eceeb196236b17998738112f37df78"
CUB_DIRECTORY_NAME = "CUB_200_2011"
CUB_IMAGE_COUNT = 11_788
CUB_TRAIN_COUNT = 5_994
CUB_TEST_COUNT = 5_794
CUB_CLASS_COUNT = 200

Split = Literal["train", "test", "all"]
ImageTransform = Callable[[Image.Image], Any]


@dataclass(frozen=True, slots=True)
class CubSample:
    """One CUB example as described by the official metadata files."""

    image_id: int
    relative_path: Path
    target: int
    is_training: bool


@dataclass(frozen=True, slots=True)
class CubValidationReport:
    """Counts produced after a successful dataset integrity check."""

    root: Path
    image_count: int
    train_count: int
    test_count: int
    class_count: int
    checked_image_files: bool


def _file_digest(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_indexed_strings(path: Path) -> dict[int, str]:
    values: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split(maxsplit=1)
            if len(parts) != 2:
                raise RuntimeError(f"Malformed metadata at {path}:{line_number}")
            index = int(parts[0])
            if index in values:
                raise RuntimeError(f"Duplicate image id {index} in {path}")
            values[index] = parts[1]
    return values


def _read_indexed_ints(path: Path) -> dict[int, int]:
    return {index: int(value) for index, value in _read_indexed_strings(path).items()}


def _load_class_names(root: Path) -> tuple[str, ...]:
    """Read the authoritative 1-based CUB label vocabulary in class-id order."""

    path = root / CUB_DIRECTORY_NAME / "classes.txt"
    indexed = _read_indexed_strings(path)
    expected_ids = set(range(1, CUB_CLASS_COUNT + 1))
    if set(indexed) != expected_ids:
        raise RuntimeError("classes.txt does not contain the 200 expected ids")

    names: list[str] = []
    for class_id in sorted(expected_ids):
        value = indexed[class_id]
        prefix, separator, raw_name = value.partition(".")
        if separator != "." or prefix != f"{class_id:03d}" or not raw_name:
            raise RuntimeError(f"Invalid class entry for id {class_id} in classes.txt: {value!r}")
        names.append(raw_name.replace("_", " "))
    return tuple(names)


def canonical_class_name(value: str) -> str:
    """Normalise harmless CUB/FuDD punctuation differences for alignment checks."""

    return "".join(character for character in value.casefold() if character.isalnum())


def validate_class_name_alignment(
    cub_class_names: Sequence[str],
    prompt_class_names: Sequence[str],
) -> None:
    """Fail closed unless CUB label ids and FuDD prompt ids name the same classes."""

    if len(cub_class_names) != CUB_CLASS_COUNT:
        raise ValueError(f"Expected {CUB_CLASS_COUNT} CUB class names")
    if len(prompt_class_names) != CUB_CLASS_COUNT:
        raise ValueError(f"Expected {CUB_CLASS_COUNT} FuDD class names")
    for class_id, (cub_name, prompt_name) in enumerate(
        zip(cub_class_names, prompt_class_names, strict=True)
    ):
        if canonical_class_name(cub_name) != canonical_class_name(prompt_name):
            raise ValueError(
                "CUB/FuDD class order mismatch at zero-based id "
                f"{class_id}: {cub_name!r} != {prompt_name!r}"
            )


def _load_samples(root: Path) -> tuple[CubSample, ...]:
    dataset_root = root / CUB_DIRECTORY_NAME
    images = _read_indexed_strings(dataset_root / "images.txt")
    labels = _read_indexed_ints(dataset_root / "image_class_labels.txt")
    splits = _read_indexed_ints(dataset_root / "train_test_split.txt")

    expected_ids = set(range(1, CUB_IMAGE_COUNT + 1))
    if set(images) != expected_ids:
        raise RuntimeError("images.txt does not contain the 11,788 expected ids")
    if set(labels) != expected_ids or set(splits) != expected_ids:
        raise RuntimeError("CUB metadata files do not refer to the same image ids")

    samples: list[CubSample] = []
    for image_id in sorted(expected_ids):
        raw_path = Path(images[image_id])
        if raw_path.is_absolute() or ".." in raw_path.parts:
            raise RuntimeError(f"Unsafe CUB image path: {raw_path}")
        class_id = labels[image_id]
        split_value = splits[image_id]
        if not 1 <= class_id <= CUB_CLASS_COUNT:
            raise RuntimeError(f"Invalid class id {class_id} for image {image_id}")
        if split_value not in (0, 1):
            raise RuntimeError(f"Invalid split value for image {image_id}")
        samples.append(
            CubSample(
                image_id=image_id,
                relative_path=raw_path,
                target=class_id - 1,
                is_training=bool(split_value),
            )
        )
    return tuple(samples)


def validate_cub(
    root: Path | str,
    *,
    verify_images: bool = True,
) -> CubValidationReport:
    """Validate official metadata, split sizes, labels, and optionally images.

    Raises:
        FileNotFoundError: If an expected metadata or image file is absent.
        RuntimeError: If metadata values or official counts are inconsistent.
    """

    root_path = Path(root).expanduser()
    dataset_root = root_path / CUB_DIRECTORY_NAME
    required_metadata = (
        "classes.txt",
        "images.txt",
        "image_class_labels.txt",
        "train_test_split.txt",
    )
    for filename in required_metadata:
        path = dataset_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing CUB metadata file: {path}")

    class_names = _load_class_names(root_path)
    samples = _load_samples(root_path)
    train_count = sum(sample.is_training for sample in samples)
    test_count = len(samples) - train_count
    labelled_class_count = len({sample.target for sample in samples})
    if train_count != CUB_TRAIN_COUNT or test_count != CUB_TEST_COUNT:
        raise RuntimeError(f"Unexpected CUB split sizes: train={train_count}, test={test_count}")
    if labelled_class_count != CUB_CLASS_COUNT:
        raise RuntimeError(f"Expected 200 labelled CUB classes, found {labelled_class_count}")

    if verify_images:
        image_root = dataset_root / "images"
        for sample in samples:
            path = image_root / sample.relative_path
            if not path.is_file():
                raise FileNotFoundError(f"Missing CUB image: {path}")

    return CubValidationReport(
        root=root_path,
        image_count=len(samples),
        train_count=train_count,
        test_count=test_count,
        class_count=len(class_names),
        checked_image_files=verify_images,
    )


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ttvr-fudd-reproduction/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            member_path = (destination / member.name).resolve()
            if os.path.commonpath((destination_resolved, member_path)) != str(destination_resolved):
                raise RuntimeError(f"Unsafe path in CUB archive: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Links are not allowed in CUB archive: {member.name}")
        tar.extractall(path=destination, members=members)


def download_cub(
    root: Path | str,
    *,
    verify_images: bool = True,
) -> CubValidationReport:
    """Download and safely extract the official CUB-200-2011 archive.

    A valid extracted dataset is reused immediately.  The archive is checked
    against the MD5 published in the official FuDD implementation before it is
    extracted.
    """

    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    try:
        return validate_cub(root_path, verify_images=verify_images)
    except (FileNotFoundError, RuntimeError):
        pass

    archive = root_path / CUB_ARCHIVE_NAME
    if not archive.is_file() or _file_digest(archive) != CUB_ARCHIVE_MD5:
        _download_file(CUB_ARCHIVE_URL, archive)
    actual_md5 = _file_digest(archive)
    if actual_md5 != CUB_ARCHIVE_MD5:
        raise RuntimeError(
            f"CUB archive checksum mismatch: expected {CUB_ARCHIVE_MD5}, found {actual_md5}"
        )

    _safe_extract(archive, root_path)
    return validate_cub(root_path, verify_images=verify_images)


class CUB200Dataset(Dataset[tuple[Any, int]]):
    """PyTorch dataset for an official CUB split.

    ``split='test'`` is the split used for all FuDD CUB results.  ``transform``
    should normally be ``CLIPBackend.preprocess``.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        split: Split = "test",
        transform: ImageTransform | None = None,
        verify_images: bool = True,
    ) -> None:
        if split not in ("train", "test", "all"):
            raise ValueError("split must be 'train', 'test', or 'all'")
        self.root = Path(root).expanduser()
        self.split = split
        self.transform = transform
        validate_cub(self.root, verify_images=verify_images)
        self.class_names = _load_class_names(self.root)

        samples = _load_samples(self.root)
        if split == "train":
            samples = tuple(sample for sample in samples if sample.is_training)
        elif split == "test":
            samples = tuple(sample for sample in samples if not sample.is_training)
        self.samples: Sequence[CubSample] = samples

    @property
    def targets(self) -> tuple[int, ...]:
        """Zero-based class labels in dataset order."""

        return tuple(sample.target for sample in self.samples)

    @property
    def fingerprint(self) -> str:
        """Stable split fingerprint used to namespace feature caches."""

        digest = hashlib.sha256(self.split.encode())
        for class_id, class_name in enumerate(self.class_names):
            digest.update(f"class:{class_id}:{class_name}\n".encode())
        for sample in self.samples:
            digest.update(f"{sample.image_id}:{sample.relative_path}:{sample.target}\n".encode())
        return digest.hexdigest()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        sample = self.samples[index]
        image_path = self.root / CUB_DIRECTORY_NAME / "images" / sample.relative_path
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.target


def prepare_cub(
    root: Path | str,
    *,
    transform: ImageTransform | None = None,
    download: bool = True,
    verify_images: bool = True,
    split: Split = "test",
) -> CUB200Dataset:
    """Prepare CUB and return an explicit dataset object.

    Pass the model preprocessing transform explicitly to keep the backend and
    input resolution coupled.
    """

    if download:
        download_cub(root, verify_images=verify_images)
    else:
        validate_cub(root, verify_images=verify_images)
    return CUB200Dataset(
        root,
        split=split,
        transform=transform,
        verify_images=False,
    )
