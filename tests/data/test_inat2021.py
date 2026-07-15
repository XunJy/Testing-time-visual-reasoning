from __future__ import annotations

import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import pytest

from ttvr.data.inat2021 import (
    INAT2021_BIRDNET_TAXONOMY_ALIGNMENT_PROTOCOL,
    INAT2021_BIRDNET_TAXONOMY_AUTHORITY_URL,
    INAT2021_BIRDNET_TAXONOMY_OVERRIDES,
    _birdnet_indices,
    _extract_missing_archive_members,
    _load_annotation_payload,
    _normalise_name,
    _resolve_taxon,
    _stable_split,
    _taxonomy_alignment_metadata,
)


def _write_birdnet_csv(path: Path, rows: list[str]) -> None:
    path.write_text(
        "birdnet_id,scientific_name,common_name,taxon_group,record_type,"
        "scientific_name_aliases,common_name_alt,common_name_aliases\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_annotation_payload_is_read_from_archive_not_an_unlocked_side_file(
    tmp_path: Path,
) -> None:
    payload = {"annotations": [], "categories": [], "images": [], "licenses": []}
    encoded = json.dumps(payload).encode()
    archive_path = tmp_path / "train_mini.json.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        member = tarfile.TarInfo("train_mini.json")
        member.size = len(encoded)
        archive.addfile(member, io.BytesIO(encoded))

    assert _load_annotation_payload(archive_path) == payload


def test_annotation_payload_requires_expected_member_and_schema(tmp_path: Path) -> None:
    wrong_member = tmp_path / "wrong-member.tar.gz"
    with tarfile.open(wrong_member, mode="w:gz") as archive:
        member = tarfile.TarInfo("other.json")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"{}"))
    with pytest.raises(RuntimeError, match="missing train_mini.json"):
        _load_annotation_payload(wrong_member)

    wrong_schema = tmp_path / "wrong-schema.tar.gz"
    with tarfile.open(wrong_schema, mode="w:gz") as archive:
        member = tarfile.TarInfo("train_mini.json")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"{}"))
    with pytest.raises(RuntimeError, match="missing keys"):
        _load_annotation_payload(wrong_schema)


def test_extract_missing_archive_members_preserves_existing_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    archive_path = root / "images.tar.gz"
    members = {
        "train_mini/species/existing.jpg": b"archive version",
        "train_mini/species/missing.jpg": b"new image",
    }
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for name, payload in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    existing = root / "train_mini/species/existing.jpg"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"keep this verified file")

    extracted = _extract_missing_archive_members(
        archive_path,
        root,
        list(members),
    )

    assert extracted == 1
    assert existing.read_bytes() == b"keep this verified file"
    assert (root / "train_mini/species/missing.jpg").read_bytes() == b"new image"
    assert _extract_missing_archive_members(archive_path, root, list(members)) == 0


def test_stable_split_is_deterministic_and_balanced_within_source_categories() -> None:
    images = [
        {"id": category * 10 + offset, "file_name": f"{category}/{offset}.jpg"}
        for category in (1, 2)
        for offset in range(6)
    ]
    annotations = {
        int(image["id"]): {
            "image_id": image["id"],
            "category_id": int(image["id"]) // 10,
        }
        for image in images
    }

    first = _stable_split(images, annotations, validation_per_class=2)
    second = _stable_split(list(reversed(images)), annotations, validation_per_class=2)
    validation_counts = Counter(
        annotations[image_id]["category_id"]
        for image_id, split in first.items()
        if split == "validation"
    )

    assert first == second
    assert validation_counts == {1: 2, 2: 2}


def test_stable_split_rejects_classes_without_training_examples() -> None:
    images = [{"id": index, "file_name": f"{index}.jpg"} for index in range(3)]
    annotations = {index: {"image_id": index, "category_id": 1} for index in range(3)}

    with pytest.raises(RuntimeError, match="too few Mini images"):
        _stable_split(images, annotations, validation_per_class=3)


def test_taxon_resolution_prefers_scientific_then_common_name_then_source_id() -> None:
    rows = {
        "BN1": {"scientific_name": "Accepted bird", "common_name": "Accepted Bird"},
        "BN2": {"scientific_name": "Renamed bird", "common_name": "Blue Bird"},
    }
    scientific = {"old bird": {"BN1"}}
    common = {"bluebird": {"BN2"}}

    exact = _resolve_taxon(
        {"id": 11, "name": "Old bird", "common_name": "Wrong common"},
        scientific,
        common,
        rows,
    )
    renamed = _resolve_taxon(
        {"id": 12, "name": "Unmatched bird", "common_name": "Blue-bird"},
        scientific,
        common,
        rows,
    )
    source = _resolve_taxon(
        {"id": 13, "name": "Source bird", "common_name": "Source Bird"},
        scientific,
        common,
        rows,
    )

    assert exact.taxon_id == "birdnet:BN1"
    assert renamed.taxon_id == "birdnet:BN2"
    assert source.taxon_id == "inat2021:13"
    assert _normalise_name("Blue-bird's  Name") == "bluebirdsname"


def test_reviewed_inat_taxonomy_overrides_resolve_all_six_to_birdnet() -> None:
    rows = {
        override.birdnet_id: {
            "scientific_name": override.accepted_scientific_name,
            "common_name": override.accepted_common_name,
        }
        for override in INAT2021_BIRDNET_TAXONOMY_OVERRIDES
    }

    resolved = {
        override.source_scientific_name: _resolve_taxon(
            {
                "id": index,
                "name": override.source_scientific_name,
                "common_name": override.source_common_name,
            },
            {},
            {},
            rows,
        )
        for index, override in enumerate(INAT2021_BIRDNET_TAXONOMY_OVERRIDES, start=1)
    }

    assert len(resolved) == 6
    assert resolved["Corvus caurinus"].taxon_id == "birdnet:BN03722"
    assert resolved["Parus minor"].taxon_id == "birdnet:BN10706"
    assert {taxon.taxon_id for taxon in resolved.values()} == {
        f"birdnet:{override.birdnet_id}" for override in INAT2021_BIRDNET_TAXONOMY_OVERRIDES
    }


def test_reviewed_inat_taxonomy_override_metadata_is_auditable_and_locked() -> None:
    metadata = _taxonomy_alignment_metadata()

    assert metadata["protocol"] == INAT2021_BIRDNET_TAXONOMY_ALIGNMENT_PROTOCOL
    assert metadata["authority"] == "AviList v2025b"
    assert metadata["authority_url"] == INAT2021_BIRDNET_TAXONOMY_AUTHORITY_URL
    assert metadata["override_count"] == 6
    assert metadata["overrides"] == [
        override.to_dict() for override in INAT2021_BIRDNET_TAXONOMY_OVERRIDES
    ]
    assert all(row["authority_record"] for row in metadata["overrides"])


def test_reviewed_inat_taxonomy_override_fails_if_birdnet_identity_changes() -> None:
    override = next(
        row
        for row in INAT2021_BIRDNET_TAXONOMY_OVERRIDES
        if row.source_scientific_name == "Corvus caurinus"
    )
    category = {
        "id": 3750,
        "name": override.source_scientific_name,
        "common_name": override.source_common_name,
    }

    with pytest.raises(RuntimeError, match="missing from BirdNET"):
        _resolve_taxon(category, {}, {}, {})
    with pytest.raises(RuntimeError, match="changed in BirdNET"):
        _resolve_taxon(
            category,
            {},
            {},
            {
                override.birdnet_id: {
                    "scientific_name": "Corvus changed",
                    "common_name": "American Crow",
                }
            },
        )


def test_birdnet_taxonomy_indices_include_aliases_and_reject_duplicate_ids(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "birdnet.csv"
    _write_birdnet_csv(
        csv_path,
        ["BN1,Accepted bird,Accepted Bird,Aves,species,Old bird,Alt Bird,Alias Bird"],
    )

    scientific, common, rows = _birdnet_indices(csv_path)

    assert scientific["old bird"] == {"BN1"}
    assert common["aliasbird"] == {"BN1"}
    assert rows["BN1"]["scientific_name"] == "Accepted bird"

    _write_birdnet_csv(
        csv_path,
        [
            "BN1,Accepted bird,Accepted Bird,Aves,species,,,",
            "BN1,Other bird,Other Bird,Aves,species,,,,",
        ],
    )
    with pytest.raises(RuntimeError, match="duplicate id"):
        _birdnet_indices(csv_path)
