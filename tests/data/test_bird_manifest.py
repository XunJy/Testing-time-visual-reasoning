from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
from PIL import Image

from ttvr.data.bird_manifest import (
    BirdSample,
    BirdTaxon,
    ManifestBirdDataset,
    load_samples,
    load_taxa,
    perceptual_dhash,
    sha256_file,
    validate_manifest,
    write_json,
    write_jsonl,
)


def _taxon(taxon_id: str = "test:1") -> BirdTaxon:
    return BirdTaxon(
        taxon_id=taxon_id,
        scientific_name="Avis exemplaris",
        common_name="Example bird",
        taxonomy_source="Test taxonomy",
        taxonomy_version="v1",
    )


def _sample(relative_path: str, digest: str, *, sample_id: str = "image-1") -> BirdSample:
    return BirdSample(
        dataset_id="test-birds",
        source_sample_id=sample_id,
        source_split="train",
        relative_path=relative_path,
        image_uri="https://example.invalid/image.jpg",
        group_id=f"group:{sample_id}",
        raw_label="Avis exemplaris",
        taxon_id="test:1",
        sha256=digest,
        phash="",
        license="CC BY 4.0",
        author="Test author",
        source="Synthetic test fixture",
    )


def test_manifest_round_trip_validation_and_dataset_view(tmp_path: Path) -> None:
    image_path = tmp_path / "images/example.png"
    image_path.parent.mkdir()
    Image.new("RGB", (12, 8), color=(10, 20, 30)).save(image_path)
    sample = _sample("images/example.png", sha256_file(image_path))
    taxon = _taxon()
    sample_path = write_jsonl(tmp_path / "samples.jsonl", [asdict(sample)])
    taxon_path = write_jsonl(tmp_path / "taxa.jsonl", [asdict(taxon)])

    samples = load_samples(sample_path)
    taxa = load_taxa(taxon_path)
    validation = validate_manifest(
        tmp_path,
        samples,
        taxa,
        dataset_id="test-birds",
        verify_images=True,
    )
    dataset = ManifestBirdDataset(tmp_path, samples, taxa, verify_images=True)

    assert validation.sample_count == 1
    assert validation.class_count == 1
    assert validation.split_counts == {"train": 1}
    assert validation.checked_images
    assert len(validation.fingerprint) == 64
    assert dataset.fingerprint == validation.fingerprint
    image, target = dataset[0]
    assert image.mode == "RGB"
    assert target == 0


def test_validation_detects_changed_image_bytes(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), color="red").save(image_path)
    sample = _sample("image.png", sha256_file(image_path))
    Image.new("RGB", (4, 4), color="blue").save(image_path)

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        validate_manifest(
            tmp_path,
            [sample],
            [_taxon()],
            dataset_id="test-birds",
            verify_images=True,
        )


def test_manifest_rejects_parent_paths_and_symlink_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative_path is unsafe"):
        _sample("../outside.png", "0" * 64)

    root = tmp_path / "dataset"
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (3, 3)).save(outside)
    link = image_dir / "linked.png"
    link.symlink_to(outside)
    sample = _sample("images/linked.png", sha256_file(outside))

    with pytest.raises(ValueError, match="escapes dataset root"):
        validate_manifest(
            root,
            [sample],
            [_taxon()],
            dataset_id="test-birds",
            verify_images=True,
        )


def test_duplicate_taxon_and_sample_ids_are_rejected(tmp_path: Path) -> None:
    sample = _sample("image.png", "0" * 64)
    with pytest.raises(ValueError, match="taxon ids must be unique"):
        validate_manifest(
            tmp_path,
            [sample],
            [_taxon(), _taxon()],
            dataset_id="test-birds",
            verify_images=False,
        )
    with pytest.raises(ValueError, match="sample ids must be unique"):
        validate_manifest(
            tmp_path,
            [sample, sample],
            [_taxon()],
            dataset_id="test-birds",
            verify_images=False,
        )


def test_json_writers_are_atomic_non_overwriting_artifacts(tmp_path: Path) -> None:
    json_path = write_json(tmp_path / "source.json", {"dataset": "birds"})
    jsonl_path = write_jsonl(tmp_path / "rows.jsonl", [{"row": 1}])

    assert json_path.read_text(encoding="utf-8").endswith("\n")
    assert jsonl_path.read_text(encoding="utf-8") == '{"row":1}\n'
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_json(json_path, {"dataset": "replacement"})
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_jsonl(jsonl_path, [{"row": 2}])


def test_perceptual_hash_is_deterministic_and_sensitive_to_direction() -> None:
    increasing = Image.new("L", (9, 8))
    decreasing = Image.new("L", (9, 8))
    increasing.putdata([column * 20 for _row in range(8) for column in range(9)])
    decreasing.putdata([(8 - column) * 20 for _row in range(8) for column in range(9)])

    first = perceptual_dhash(increasing)
    second = perceptual_dhash(increasing.copy())

    assert first == second == "0000000000000000"
    assert perceptual_dhash(decreasing) == "ffffffffffffffff"

