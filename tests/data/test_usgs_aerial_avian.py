from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from ttvr.data.bird_manifest import load_samples, load_taxa, validate_manifest
from ttvr.data.bird_source_archive import validated_zip_members
from ttvr.data.usgs_aerial_avian import (
    _EXPECTED_LABELMAP,
    USGS_AERIAL_AVIAN_DATASET_ID,
    audit_usgs_aerial_archives,
    audit_usgs_aerial_strict_cub,
    plan_usgs_aerial_avian,
    prepare_usgs_aerial_avian,
)


def _birdnet_csv(path: Path) -> Path:
    path.write_text(
        "birdnet_id,scientific_name,common_name,taxon_group,record_type\n"
        "BN14015,Somateria mollissima,Common Eider,Aves,species\n"
        "BN03328,Clangula hyemalis,Long-tailed Duck,Aves,species\n"
        "BN08586,Melanitta americana,Black Scoter,Aves,species\n"
        "BN08587,Melanitta deglandi,White-winged Scoter,Aves,species\n",
        encoding="utf-8",
    )
    return path


def _png_bytes(colour: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (12, 8), colour).save(output, format="PNG")
    return output.getvalue()


def _write_archives(tmp_path: Path) -> tuple[Path, Path, Path, list[str]]:
    labelmap = "\n".join(
        repr({"id": category_id, "name": name, "supercategory": ""})
        for category_id, name in _EXPECTED_LABELMAP.items()
    )
    rows = {
        "annotations/train.txt": [
            "images/train/1/frame_a_1.png 0",
            "images/train/3/frame_a_2.png 2",
            "images/train/4/frame_b_1.png 3",
        ],
        "annotations/val.txt": [
            "images/eval/2/frame_c_1.png 1",
            "images/eval/5/frame_c_2.png 4",
        ],
        "annotations/test.txt": ["images/test/6/frame_d_1.png 5"],
    }
    annotations = tmp_path / "annotations.zip"
    with zipfile.ZipFile(annotations, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("annotations/labelmap.txt", labelmap + "\n")
        for name, values in rows.items():
            bundle.writestr(name, "\n".join(values) + "\n")
    image_paths = [line.rsplit(" ", 1)[0] for values in rows.values() for line in values]
    images = tmp_path / "images.zip"
    with zipfile.ZipFile(images, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for index, name in enumerate(image_paths):
            bundle.writestr(name, _png_bytes((20 * index, 80, 180)))
    return annotations, images, _birdnet_csv(tmp_path / "birdnet.csv"), image_paths


def test_plan_and_prepare_publisher_crops_without_recropping(tmp_path: Path) -> None:
    annotations, images, birdnet, _paths = _write_archives(tmp_path)
    plan = plan_usgs_aerial_avian(
        annotations,
        birdnet_csv_path=birdnet,
        require_official_lock=False,
    )

    assert len(plan.all_rows) == 6
    assert len(plan.specific_rows) == len(plan.taxa) == 4
    assert plan.specific_split_counts == {"test": 1, "train": 2, "validation": 1}
    assert {row.raw_label for row in plan.specific_rows} == {
        "Common Eider",
        "Long-tailed Duck",
        "Black Scoter",
        "White-winged Scoter",
    }

    root = tmp_path / "prepared"
    report = prepare_usgs_aerial_avian(
        root,
        birdnet_csv_path=birdnet,
        annotations_path=annotations,
        images_path=images,
        require_official_lock=False,
    )

    assert report.publisher_crop_count == report.extracted_crops == 4
    assert report.omitted_coarse_crops == 2
    samples = load_samples(root / "manifests/samples.jsonl")
    taxa = load_taxa(root / "manifests/taxa.jsonl")
    validation = validate_manifest(
        root,
        samples,
        taxa,
        dataset_id=USGS_AERIAL_AVIAN_DATASET_ID,
        verify_images=True,
    )
    assert validation.fingerprint == report.manifest_fingerprint
    assert all(sample.relative_path.startswith("images/") for sample in samples)
    assert {sample.source_split for sample in samples} == {"train", "validation", "test"}
    provenance = [
        json.loads(line)
        for line in (root / "manifests/crop_provenance.jsonl").read_text().splitlines()
    ]
    assert all(row["local_crop_applied"] is False for row in provenance)
    assert all(row["bbox_official"] is None for row in provenance)
    assert all("no source bbox released" in row["crop_provenance"] for row in provenance)
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        prepare_usgs_aerial_avian(
            root,
            birdnet_csv_path=birdnet,
            annotations_path=annotations,
            images_path=images,
            require_official_lock=False,
        )


def test_strict_cub_audit_is_non_mutating(tmp_path: Path) -> None:
    annotations, _images, birdnet, _paths = _write_archives(tmp_path)
    plan = plan_usgs_aerial_avian(
        annotations,
        birdnet_csv_path=birdnet,
        require_official_lock=False,
    )

    audit = audit_usgs_aerial_strict_cub(plan.specific_rows, ["BN14015", "BN08587"])

    assert audit.total_crops == audit.total_taxa == 4
    assert audit.excluded_crops == audit.excluded_taxa == 2
    assert audit.retained_crops == audit.retained_taxa == 2
    assert len(plan.specific_rows) == 4


def test_archive_pairing_and_unsafe_annotation_paths_fail_closed(tmp_path: Path) -> None:
    annotations, images, birdnet, image_paths = _write_archives(tmp_path)
    plan = plan_usgs_aerial_avian(
        annotations,
        birdnet_csv_path=birdnet,
        require_official_lock=False,
    )
    wrong = tmp_path / "wrong-images.zip"
    with zipfile.ZipFile(wrong, "w") as bundle:
        for name in image_paths[:-1]:
            bundle.writestr(name, _png_bytes((0, 0, 0)))
    with pytest.raises(RuntimeError, match="file-set mismatch"):
        audit_usgs_aerial_archives(plan, wrong, require_official_lock=False)

    malicious = tmp_path / "malicious-annotations.zip"
    labelmap = "\n".join(
        repr({"id": key, "name": value, "supercategory": ""})
        for key, value in _EXPECTED_LABELMAP.items()
    )
    with zipfile.ZipFile(malicious, "w") as bundle:
        bundle.writestr("annotations/labelmap.txt", labelmap)
        bundle.writestr("annotations/train.txt", "../escape.png 0\n")
        bundle.writestr("annotations/val.txt", "images/eval/2/frame_1.png 1\n")
        bundle.writestr(
            "annotations/test.txt",
            "images/test/3/frame_2.png 2\n"
            "images/test/4/frame_3.png 3\n"
            "images/test/5/frame_4.png 4\n"
            "images/test/6/frame_5.png 5\n",
        )
    with pytest.raises(ValueError, match="unsafe relative path"):
        plan_usgs_aerial_avian(
            malicious,
            birdnet_csv_path=birdnet,
            require_official_lock=False,
        )


def test_zip_symlink_member_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("images/link.png")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(info, "target")

    with pytest.raises(RuntimeError, match="symbolic link"):
        validated_zip_members(archive)


def test_official_lock_rejects_synthetic_annotations(tmp_path: Path) -> None:
    annotations, _images, birdnet, _paths = _write_archives(tmp_path)
    with pytest.raises(RuntimeError, match="official release"):
        plan_usgs_aerial_avian(annotations, birdnet_csv_path=birdnet)
