from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
import zlib
from pathlib import Path

import pytest
from PIL import Image

import ttvr.data.nm_uas_waterfowl as nm
from ttvr.data.bird_manifest import load_samples, load_taxa, validate_manifest
from ttvr.data.nm_uas_waterfowl import (
    NM_UAS_WATERFOWL_DATASET_ID,
    NM_UAS_WATERFOWL_EMBEDDED_LICENSE,
    NM_UAS_WATERFOWL_LICENSE,
    audit_nm_uas_archive,
    audit_nm_uas_strict_cub,
    plan_nm_uas_waterfowl,
    prepare_nm_uas_waterfowl,
)


def _birdnet_csv(path: Path) -> Path:
    path.write_text(
        "birdnet_id,scientific_name,common_name,taxon_group,record_type\n"
        "BN01898,Branta canadensis,Canada Goose,Aves,species\n"
        "BN00949,Antigone canadensis,Sandhill Crane,Aves,species\n"
        "BN00713,Anas platyrhynchos,Mallard,Aves,species\n"
        "BN00691,Anas acuta,Northern Pintail,Aves,species\n"
        "BN08400,Mareca americana,American Wigeon,Aves,species\n"
        "BN08405,Mareca strepera,Gadwall,Aves,species\n"
        "BN14030,Spatula clypeata,Northern Shoveler,Aves,species\n",
        encoding="utf-8",
    )
    return path


def _metadata() -> dict[str, object]:
    categories = [
        {"id": category_id, "name": name, "supercategory": "Bird"}
        for category_id, name in nm._EXPECTED_CATEGORY_NAMES.items()
    ]
    annotations = [
        {
            "id": category_id,
            "image_id": 1,
            "category_id": category_id,
            "bbox": [float(category_id), 2.0, 3.0, 4.0],
            "iscrowd": 0,
        }
        for category_id in nm._EXPECTED_CATEGORY_NAMES
    ]
    return {
        "info": {"description": "synthetic consensus fixture"},
        "images": [
            {
                "id": 1,
                "file_name": "BDA_fixture_1.JPG",
                "license": 1,
                "width": 24,
                "height": 20,
                "date_captured": "2018-11-27 11:44:53",
            }
        ],
        "categories": categories,
        "annotations": annotations,
        "licenses": {
            "id": 1,
            "url": nm.NM_UAS_WATERFOWL_EMBEDDED_LICENSE_URL,
            "name": NM_UAS_WATERFOWL_EMBEDDED_LICENSE,
        },
    }


def _jpeg_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (24, 20), (50, 120, 180)).save(output, format="JPEG", quality=90)
    return output.getvalue()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    metadata = tmp_path / "expert_refined.json"
    metadata.write_text(json.dumps(_metadata(), separators=(",", ":")), encoding="utf-8")
    archive = tmp_path / "nm.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(nm.NM_UAS_EXPERT_REFINED_MEMBER, metadata.read_bytes())
        for member in sorted(nm._OTHER_METADATA_MEMBERS):
            bundle.writestr(member, "{}")
        bundle.writestr(
            f"{nm._EXPERT_IMAGE_PREFIX}BDA_fixture_1.jpg",
            _jpeg_bytes(),
        )
    return metadata, archive, _birdnet_csv(tmp_path / "birdnet.csv")


def test_plan_prepare_and_license_discrepancy_provenance(tmp_path: Path) -> None:
    metadata, archive, birdnet = _write_fixture(tmp_path)
    plan = plan_nm_uas_waterfowl(
        metadata,
        birdnet_csv_path=birdnet,
        require_official_lock=False,
    )

    assert len(plan.images) == 1
    assert len(plan.crops) == len(plan.taxa) == 7
    assert plan.coarse_crop_count == 2
    assert plan.clipped_bbox_count == 0
    assert {crop.raw_label for crop in plan.crops}.isdisjoint({"Other", "Teal"})

    root = tmp_path / "prepared"
    report = prepare_nm_uas_waterfowl(
        root,
        birdnet_csv_path=birdnet,
        metadata_path=metadata,
        archive_path=archive,
        require_official_lock=False,
    )

    assert report.crop_count == report.written_crops == 7
    assert report.coarse_crops_omitted == 2
    assert not (root / "source_images/BDA_fixture_1.jpg").exists()
    samples = load_samples(root / "manifests/samples.jsonl")
    taxa = load_taxa(root / "manifests/taxa.jsonl")
    validation = validate_manifest(
        root,
        samples,
        taxa,
        dataset_id=NM_UAS_WATERFOWL_DATASET_ID,
        verify_images=True,
    )
    assert validation.fingerprint == report.manifest_fingerprint
    assert {sample.license for sample in samples} == {NM_UAS_WATERFOWL_LICENSE}
    source = json.loads((root / "manifests/source.json").read_text())
    assert source["landing_page_license"] == "CC BY-NC 4.0"
    assert source["embedded_coco_license"] == NM_UAS_WATERFOWL_EMBEDDED_LICENSE
    assert "Both statements are retained" in source["license_metadata_discrepancy"]
    provenance = [
        json.loads(line)
        for line in (root / "manifests/crop_provenance.jsonl").read_text().splitlines()
    ]
    assert all(row["expert_consensus_only"] for row in provenance)
    assert all(row["crop_geometry"]["recipe_id"] == "ttvr-square-context-v1" for row in provenance)
    assert all(row["bbox_xywh_official"] for row in provenance)
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        prepare_nm_uas_waterfowl(
            root,
            birdnet_csv_path=birdnet,
            metadata_path=metadata,
            archive_path=archive,
            require_official_lock=False,
        )


def test_strict_cub_retains_five_non_target_taxa(tmp_path: Path) -> None:
    metadata, _archive, birdnet = _write_fixture(tmp_path)
    plan = plan_nm_uas_waterfowl(
        metadata,
        birdnet_csv_path=birdnet,
        require_official_lock=False,
    )

    audit = audit_nm_uas_strict_cub(plan.crops, ["BN00713", "birdnet:BN08405"])

    assert audit.total_crops == audit.total_taxa == 7
    assert audit.excluded_crops == audit.excluded_taxa == 2
    assert audit.retained_crops == audit.retained_taxa == 5
    assert dict(audit.excluded_counts_by_taxon) == {
        "birdnet:BN00713": 1,
        "birdnet:BN08405": 1,
    }
    assert len(plan.crops) == 7


def test_archive_audit_requires_exact_expert_metadata_and_images(tmp_path: Path) -> None:
    metadata, archive, birdnet = _write_fixture(tmp_path)
    plan = plan_nm_uas_waterfowl(
        metadata,
        birdnet_csv_path=birdnet,
        require_official_lock=False,
    )
    audit = audit_nm_uas_archive(plan, archive, require_official_lock=False)
    assert audit.expert_image_count == 1
    assert audit.embedded_metadata_sha256 == plan.metadata_sha256

    wrong = tmp_path / "wrong.zip"
    with zipfile.ZipFile(wrong, "w") as bundle:
        bundle.writestr(nm.NM_UAS_EXPERT_REFINED_MEMBER, metadata.read_bytes())
        for member in sorted(nm._OTHER_METADATA_MEMBERS):
            bundle.writestr(member, "{}")
    with pytest.raises(RuntimeError, match="expert image members"):
        audit_nm_uas_archive(plan, wrong, require_official_lock=False)


def test_metadata_schema_bbox_and_path_fail_closed(tmp_path: Path) -> None:
    payload = _metadata()
    payload["images"][0]["file_name"] = "../escape.JPG"  # type: ignore[index]
    metadata = tmp_path / "unsafe.json"
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    birdnet = _birdnet_csv(tmp_path / "birdnet.csv")
    with pytest.raises(ValueError, match="unsafe relative path"):
        plan_nm_uas_waterfowl(
            metadata,
            birdnet_csv_path=birdnet,
            require_official_lock=False,
        )

    payload = _metadata()
    payload["annotations"][0]["bbox"] = [1, 1, -2, 3]  # type: ignore[index]
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="positive width"):
        plan_nm_uas_waterfowl(
            metadata,
            birdnet_csv_path=birdnet,
            require_official_lock=False,
        )


def test_locked_remote_member_decoder_validates_header_crc_and_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filename = "fixture/expert.json"
    content = b'{"expert":"consensus"}'
    compressor = zlib.compressobj(level=6, wbits=-zlib.MAX_WBITS)
    compressed = compressor.compress(content) + compressor.flush()
    crc = zlib.crc32(content) & 0xFFFFFFFF
    header = struct.pack(
        "<IHHHHHIIIHH",
        0x04034B50,
        20,
        0,
        zipfile.ZIP_DEFLATED,
        0,
        0,
        crc,
        len(compressed),
        len(content),
        len(filename.encode()),
        0,
    )
    payload = header + filename.encode() + compressed
    monkeypatch.setattr(nm, "NM_UAS_EXPERT_REFINED_MEMBER", filename)
    monkeypatch.setattr(nm, "NM_UAS_EXPERT_REFINED_CRC32", crc)
    monkeypatch.setattr(nm, "NM_UAS_EXPERT_REFINED_COMPRESSED_SIZE", len(compressed))
    monkeypatch.setattr(nm, "NM_UAS_EXPERT_REFINED_SIZE", len(content))
    monkeypatch.setattr(nm, "NM_UAS_EXPERT_REFINED_RANGE_SIZE", len(payload))
    monkeypatch.setattr(nm, "NM_UAS_EXPERT_REFINED_SHA256", hashlib.sha256(content).hexdigest())

    assert nm._decode_locked_metadata_range(payload) == content
    with pytest.raises(RuntimeError, match="deflate data|checksum mismatch"):
        nm._decode_locked_metadata_range(payload[:-1] + bytes([payload[-1] ^ 1]))


def test_official_lock_rejects_synthetic_metadata(tmp_path: Path) -> None:
    metadata, _archive, birdnet = _write_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="official archive member"):
        plan_nm_uas_waterfowl(metadata, birdnet_csv_path=birdnet)
