"""Locked, dataset-neutral manifests for multi-source bird experiments."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class BirdTaxon:
    """One accepted species identity shared across source datasets."""

    taxon_id: str
    scientific_name: str
    common_name: str
    taxonomy_source: str
    taxonomy_version: str

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class BirdSample:
    """One immutable source image after taxonomy and licence resolution."""

    dataset_id: str
    source_sample_id: str
    source_split: str
    relative_path: str
    image_uri: str
    group_id: str
    raw_label: str
    taxon_id: str
    sha256: str
    phash: str
    license: str
    author: str
    source: str

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_id",
            "source_sample_id",
            "source_split",
            "relative_path",
            "image_uri",
            "group_id",
            "raw_label",
            "taxon_id",
            "license",
            "author",
            "source",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"relative_path is unsafe: {self.relative_path}")
        invalid_sha = len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        )
        if invalid_sha:
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        if self.phash and (
            len(self.phash) != 16
            or any(character not in "0123456789abcdef" for character in self.phash)
        ):
            raise ValueError("phash must be empty or 16 lowercase hexadecimal characters")


@dataclass(frozen=True, slots=True)
class ManifestValidation:
    dataset_id: str
    sample_count: int
    class_count: int
    split_counts: dict[str, int]
    fingerprint: str
    checked_images: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_dhash(image: Image.Image) -> str:
    """Return a deterministic 64-bit difference hash for duplicate auditing."""

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    pixels = list(image.convert("L").resize((9, 8), resampling).getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(
                pixels[row * 9 + column] > pixels[row * 9 + column + 1]
            )
    return f"{value:016x}"


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> Path:
    """Write a new JSONL artifact atomically and refuse replacement."""

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_json(path: Path | str, value: Any) -> Path:
    """Write a new JSON artifact atomically and refuse replacement."""

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite JSON artifact: {output}")
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Malformed JSON at {source}:{line_number}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"Manifest row is not an object at {source}:{line_number}")
            rows.append(value)
    if not rows:
        raise RuntimeError(f"Manifest must not be empty: {source}")
    return rows


def load_taxa(path: Path | str) -> tuple[BirdTaxon, ...]:
    return tuple(BirdTaxon(**row) for row in _read_jsonl(path))


def load_samples(path: Path | str) -> tuple[BirdSample, ...]:
    return tuple(BirdSample(**row) for row in _read_jsonl(path))


def validate_manifest(
    root: Path | str,
    samples: Sequence[BirdSample],
    taxa: Sequence[BirdTaxon],
    *,
    dataset_id: str,
    verify_images: bool = True,
) -> ManifestValidation:
    """Validate identities, file hashes, and the locked manifest fingerprint."""

    if not dataset_id.strip() or not samples or not taxa:
        raise ValueError("dataset_id, samples, and taxa must not be empty")
    root_path = Path(root).expanduser().resolve()
    taxon_by_id = {taxon.taxon_id: taxon for taxon in taxa}
    if len(taxon_by_id) != len(taxa):
        raise ValueError("taxon ids must be unique")
    sample_ids = [sample.source_sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("source sample ids must be unique within a dataset")
    if {sample.dataset_id for sample in samples} != {dataset_id}:
        raise ValueError("all sample dataset ids must match dataset_id")
    unknown = sorted({sample.taxon_id for sample in samples} - set(taxon_by_id))
    if unknown:
        raise ValueError(f"samples refer to unknown taxa: {unknown[:5]}")

    split_counts: dict[str, int] = {}
    digest = hashlib.sha256(f"bird-manifest-v1\0{dataset_id}\n".encode())
    for taxon in sorted(taxa, key=lambda item: item.taxon_id):
        digest.update(json.dumps(asdict(taxon), sort_keys=True).encode())
        digest.update(b"\n")
    for sample in sorted(samples, key=lambda item: item.source_sample_id):
        split_counts[sample.source_split] = split_counts.get(sample.source_split, 0) + 1
        digest.update(json.dumps(asdict(sample), sort_keys=True).encode())
        digest.update(b"\n")
        if verify_images:
            image_path = _contained_path(root_path, sample.relative_path)
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing manifest image: {image_path}")
            actual = sha256_file(image_path)
            if actual != sample.sha256:
                raise RuntimeError(
                    f"Image checksum mismatch for {sample.source_sample_id}: "
                    f"expected {sample.sha256}, found {actual}"
                )
    return ManifestValidation(
        dataset_id=dataset_id,
        sample_count=len(samples),
        class_count=len({sample.taxon_id for sample in samples}),
        split_counts=dict(sorted(split_counts.items())),
        fingerprint=digest.hexdigest(),
        checked_images=verify_images,
    )


def _contained_path(root: Path, relative_path: str) -> Path:
    """Resolve a manifest path and reject symlinks that escape its dataset root."""

    resolved_root = root.resolve()
    candidate = (resolved_root / relative_path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"manifest image escapes dataset root: {relative_path}") from error
    return candidate


class ManifestBirdDataset(Dataset[tuple[Any, int]]):
    """PyTorch view over selected rows while retaining a local label space."""

    def __init__(
        self,
        root: Path | str,
        samples: Sequence[BirdSample],
        taxa: Sequence[BirdTaxon],
        *,
        transform: Any = None,
        splits: Iterable[str] | None = None,
        included_taxon_ids: Iterable[str] | None = None,
        excluded_taxon_ids: Iterable[str] = (),
        verify_images: bool = False,
    ) -> None:
        if not samples:
            raise ValueError("samples must not be empty")
        dataset_ids = {sample.dataset_id for sample in samples}
        if len(dataset_ids) != 1:
            raise ValueError("ManifestBirdDataset accepts one source dataset at a time")
        self.dataset_id = next(iter(dataset_ids))
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        taxon_by_id = {taxon.taxon_id: taxon for taxon in taxa}
        if len(taxon_by_id) != len(taxa):
            raise ValueError("taxon ids must be unique")
        allowed_splits = None if splits is None else set(splits)
        allowed_taxa = None if included_taxon_ids is None else set(included_taxon_ids)
        excluded = set(excluded_taxon_ids)
        selected = tuple(
            sample
            for sample in samples
            if (allowed_splits is None or sample.source_split in allowed_splits)
            and (allowed_taxa is None or sample.taxon_id in allowed_taxa)
            and sample.taxon_id not in excluded
        )
        if not selected:
            raise ValueError("filters removed every manifest sample")
        selected_taxa = tuple(sorted({sample.taxon_id for sample in selected}))
        missing = set(selected_taxa) - set(taxon_by_id)
        if missing:
            raise ValueError(f"selected samples have unknown taxa: {sorted(missing)[:5]}")
        self.taxa = tuple(taxon_by_id[taxon_id] for taxon_id in selected_taxa)
        self.taxon_ids = selected_taxa
        self.class_names = tuple(taxon.common_name for taxon in self.taxa)
        self.scientific_names = tuple(taxon.scientific_name for taxon in self.taxa)
        self.samples = selected
        self._target_by_taxon = {
            taxon_id: target for target, taxon_id in enumerate(self.taxon_ids)
        }
        self.targets = tuple(self._target_by_taxon[sample.taxon_id] for sample in selected)
        validation = validate_manifest(
            self.root,
            self.samples,
            self.taxa,
            dataset_id=self.dataset_id,
            verify_images=verify_images,
        )
        self.fingerprint = validation.fingerprint

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        sample = self.samples[index]
        with Image.open(_contained_path(self.root, sample.relative_path)) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]
