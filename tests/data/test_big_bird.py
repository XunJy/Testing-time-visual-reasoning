from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageOps

from ttvr.data.big_bird import (
    BIG_BIRD_DATASET_ID,
    GCSObjectRecord,
    _contained_output_path,
    _list_gcs_objects,
    _stream_download_to_path,
    audit_big_bird_strict_cub,
    download_big_bird_metadata,
    plan_big_bird,
    prepare_big_bird,
)
from ttvr.data.bird_manifest import load_samples, load_taxa, validate_manifest


def _write_birdnet_csv(path: Path, rows: list[str] | None = None) -> None:
    path.write_text(
        "birdnet_id,scientific_name,common_name,taxon_group,record_type,"
        "scientific_name_aliases\n"
        + "\n".join(rows or ["BN1,Ardea alba,Great Egret,Aves,species,"])
        + "\n",
        encoding="utf-8",
    )


def _metadata() -> dict[str, object]:
    return {
        "info": {"version": "fixture-v1", "license": "fixture"},
        "categories": [
            {"id": 0, "name": "empty"},
            {"id": 1, "name": "great egret"},
            {"id": 6, "name": "bird"},
        ],
        "images": [
            {"id": "specific.jpg", "file_name": "specific.jpg", "width": 12, "height": 10},
            {"id": "generic.jpg", "file_name": "generic.jpg", "width": 12, "height": 10},
            {"id": "empty.jpg", "file_name": "empty.jpg", "width": 12, "height": 10},
        ],
        "annotations": [
            {
                "id": "specific.jpg_ann_000",
                "image_id": "specific.jpg",
                "category_id": 1,
                "sequence_level_annotation": False,
                "bbox": [-0.5, 2.0, 4.0, 4.0],
                "genus": "ardea",
                "species": "alba",
            },
            {
                "id": "generic.jpg_ann_000",
                "image_id": "generic.jpg",
                "category_id": 6,
                "sequence_level_annotation": False,
                "bbox": [1.0, 1.0, 2.0, 2.0],
            },
            {
                "id": "empty.jpg_ann",
                "image_id": "empty.jpg",
                "category_id": 0,
                "sequence_level_annotation": False,
            },
        ],
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(_metadata()), encoding="utf-8")
    birdnet_csv = tmp_path / "birdnet.csv"
    _write_birdnet_csv(birdnet_csv)
    source = tmp_path / "publisher-specific.png"
    image = Image.new("RGB", (12, 10), color=(20, 40, 60))
    for x in range(12):
        image.putpixel((x, 3), (x * 10, 50, 100))
    image.save(source)
    return metadata_path, birdnet_csv, source


def _gcs_record(source: Path, *, md5_base64: str | None = None) -> GCSObjectRecord:
    digest = hashlib.md5(source.read_bytes()).digest()  # noqa: S324 - fixture checksum.
    return GCSObjectRecord(
        object_name="big-bird/specific.jpg",
        md5_base64=md5_base64 or base64.b64encode(digest).decode("ascii"),
        size=source.stat().st_size,
        generation="fixture-generation-1",
        content_type="image/png",
    )


def test_plan_filters_generic_and_empty_before_any_image_download(tmp_path: Path) -> None:
    metadata_path, birdnet_csv, _source = _write_fixture(tmp_path)

    plan = plan_big_bird(
        metadata_path,
        birdnet_csv_path=birdnet_csv,
        require_official_lock=False,
    )

    assert len(plan.images) == 1
    assert plan.images[0].file_name == "specific.jpg"
    assert len(plan.crops) == 1
    crop = plan.crops[0]
    assert crop.taxon_id == "birdnet:BN1"
    assert crop.bbox_xyxy_original == (-0.5, 2.0, 3.5, 6.0)
    assert crop.bbox_xyxy_effective == (0.0, 2.0, 3.5, 6.0)
    assert crop.bbox_was_clipped
    assert plan.generic_bird_crop_count == 1
    assert plan.empty_annotation_count == 1


def test_prepare_downloads_only_selected_image_and_writes_audited_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_path, birdnet_csv, source = _write_fixture(tmp_path)
    root = tmp_path / "prepared"
    requested_objects: list[set[str]] = []
    downloaded_urls: list[str] = []

    def fake_list(required: set[str]) -> dict[str, GCSObjectRecord]:
        requested_objects.append(set(required))
        return {"big-bird/specific.jpg": _gcs_record(source)}

    def fake_download(
        url: str,
        destination: Path,
        *,
        allowed_host: str,
        attempts: int = 4,
        timeout: int = 120,
    ) -> None:
        del allowed_host, attempts, timeout
        downloaded_urls.append(url)
        shutil.copyfile(source, destination)

    monkeypatch.setattr("ttvr.data.big_bird._list_gcs_objects", fake_list)
    monkeypatch.setattr("ttvr.data.big_bird._stream_download_to_path", fake_download)

    report = prepare_big_bird(
        root,
        birdnet_csv_path=birdnet_csv,
        metadata_path=metadata_path,
        workers=1,
        require_official_lock=False,
    )

    assert requested_objects == [{"big-bird/specific.jpg"}]
    assert len(downloaded_urls) == 1
    assert downloaded_urls[0].endswith("/big-bird/specific.jpg")
    assert report.source_image_count == report.downloaded_source_images == 1
    assert report.crop_count == report.written_crops == 1
    assert report.clipped_bbox_count == report.generic_bird_crops_omitted == 1
    assert not (root / "source_images/specific.jpg").exists()

    samples = load_samples(root / "manifests/samples.jsonl")
    taxa = load_taxa(root / "manifests/taxa.jsonl")
    validation = validate_manifest(
        root,
        samples,
        taxa,
        dataset_id=BIG_BIRD_DATASET_ID,
        verify_images=True,
    )
    assert validation.fingerprint == report.manifest_fingerprint
    assert samples[0].group_id == "big-bird-source-image:specific.jpg"
    assert samples[0].image_uri.endswith("/big-bird/specific.jpg")
    with Image.open(root / samples[0].relative_path) as crop:
        assert crop.width == crop.height == 5

    provenance = [
        json.loads(line)
        for line in (root / "manifests/crop_provenance.jsonl").read_text().splitlines()
    ]
    assert provenance[0]["bbox_xyxy_official"] == [-0.5, 2.0, 3.5, 6.0]
    assert provenance[0]["bbox_was_clipped"] is True
    assert provenance[0]["gcs_md5_base64"] == _gcs_record(source).md5_base64
    assert provenance[0]["crop_sha256"] == samples[0].sha256
    source_manifest = json.loads((root / "manifests/source.json").read_text())
    assert "private training artifacts" in source_manifest["distribution_policy"]
    assert source_manifest["metadata_publisher_checksum"] is None

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        prepare_big_bird(
            root,
            birdnet_csv_path=birdnet_csv,
            metadata_path=metadata_path,
            workers=1,
            require_official_lock=False,
        )


def test_prepare_applies_explicit_exif_rotation_to_metadata_bbox_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _metadata()
    payload["images"][0].update(width=40, height=60)  # type: ignore[index]
    payload["annotations"][0]["bbox"] = [20.0, 40.0, 20.0, 20.0]  # type: ignore[index]
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    birdnet_csv = tmp_path / "birdnet.csv"
    _write_birdnet_csv(birdnet_csv)

    source = tmp_path / "publisher-oriented.jpg"
    raw = Image.new("RGB", (60, 40), color=(180, 20, 20))
    for x in range(40, 60):
        for y in range(20):
            raw.putpixel((x, y), (20, 180, 20))
    exif = Image.Exif()
    exif[274] = 6
    raw.save(source, format="JPEG", quality=100, subsampling=0, exif=exif)

    monkeypatch.setattr(
        "ttvr.data.big_bird._list_gcs_objects",
        lambda _required: {"big-bird/specific.jpg": _gcs_record(source)},
    )
    monkeypatch.setattr(
        "ttvr.data.big_bird._stream_download_to_path",
        lambda _url, destination, **_kwargs: shutil.copyfile(source, destination),
    )
    root = tmp_path / "prepared"
    report = prepare_big_bird(
        root,
        birdnet_csv_path=birdnet_csv,
        metadata_path=metadata_path,
        workers=1,
        context_scale=1.0,
        require_official_lock=False,
    )

    assert report.exif_transposed_source_images == 1
    samples = load_samples(root / "manifests/samples.jsonl")
    with Image.open(source) as encoded:
        expected = ImageOps.exif_transpose(encoded).convert("RGB").crop((20, 40, 40, 60))
    with Image.open(root / samples[0].relative_path) as actual:
        assert actual.size == expected.size == (20, 20)
        assert actual.convert("RGB").tobytes() == expected.tobytes()

    provenance = json.loads((root / "manifests/crop_provenance.jsonl").read_text().splitlines()[0])
    assert provenance["bbox_coordinate_space"] == "official metadata display orientation"
    assert provenance["source_decoded_size"] == [60, 40]
    assert provenance["source_effective_size"] == [40, 60]
    assert provenance["source_exif_orientation"] == 6
    assert provenance["source_exif_transpose_applied"] is True
    source_image = json.loads((root / "manifests/source_images.jsonl").read_text().splitlines()[0])
    assert source_image["exif_orientation"] == 6
    assert source_image["exif_transpose_applied"] is True
    source_manifest = json.loads((root / "manifests/source.json").read_text())
    assert source_manifest["report"]["exif_transposed_source_images"] == 1
    assert (
        "every other dimension mismatch fails closed" in source_manifest["image_orientation_policy"]
    )


@pytest.mark.parametrize(
    ("decoded_size", "orientation"),
    [((10, 12), None), ((10, 12), 3), ((11, 10), 6)],
)
def test_other_dimensions_without_matching_axis_swapping_exif_still_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decoded_size: tuple[int, int],
    orientation: int | None,
) -> None:
    metadata_path, birdnet_csv, _source = _write_fixture(tmp_path)
    source = tmp_path / f"mismatch-{decoded_size}-{orientation}.jpg"
    image = Image.new("RGB", decoded_size, color=(20, 40, 60))
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    image.save(source, format="JPEG", exif=exif)
    monkeypatch.setattr(
        "ttvr.data.big_bird._list_gcs_objects",
        lambda _required: {"big-bird/specific.jpg": _gcs_record(source)},
    )
    monkeypatch.setattr(
        "ttvr.data.big_bird._stream_download_to_path",
        lambda _url, destination, **_kwargs: shutil.copyfile(source, destination),
    )

    with pytest.raises(RuntimeError, match="source dimensions changed"):
        prepare_big_bird(
            tmp_path / "prepared",
            birdnet_csv_path=birdnet_csv,
            metadata_path=metadata_path,
            workers=1,
            require_official_lock=False,
        )


def test_source_object_md5_mismatch_fails_before_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_path, birdnet_csv, source = _write_fixture(tmp_path)
    root = tmp_path / "prepared"
    wrong_md5 = base64.b64encode(b"x" * 16).decode("ascii")
    monkeypatch.setattr(
        "ttvr.data.big_bird._list_gcs_objects",
        lambda _required: {"big-bird/specific.jpg": _gcs_record(source, md5_base64=wrong_md5)},
    )
    monkeypatch.setattr(
        "ttvr.data.big_bird._stream_download_to_path",
        lambda _url, destination, **_kwargs: shutil.copyfile(source, destination),
    )

    with pytest.raises(RuntimeError, match="source MD5 mismatch"):
        prepare_big_bird(
            root,
            birdnet_csv_path=birdnet_csv,
            metadata_path=metadata_path,
            workers=1,
            require_official_lock=False,
        )

    assert not (root / "manifests").exists()
    assert not (root / "crops").exists()


def test_taxonomy_ambiguity_and_unreviewed_absence_fail_closed(tmp_path: Path) -> None:
    metadata_path, birdnet_csv, _source = _write_fixture(tmp_path)
    _write_birdnet_csv(
        birdnet_csv,
        [
            "BN1,Ardea first,First Egret,Aves,species,Ardea alba",
            "BN2,Ardea second,Second Egret,Aves,species,Ardea alba",
        ],
    )
    with pytest.raises(RuntimeError, match="exactly one BirdNET"):
        plan_big_bird(
            metadata_path,
            birdnet_csv_path=birdnet_csv,
            require_official_lock=False,
        )

    _write_birdnet_csv(birdnet_csv, ["BN3,Other bird,Other Bird,Aves,species,"])
    with pytest.raises(RuntimeError, match="resolved to 0"):
        plan_big_bird(
            metadata_path,
            birdnet_csv_path=birdnet_csv,
            require_official_lock=False,
        )


def test_reviewed_antarctic_shag_override_is_version_locked(tmp_path: Path) -> None:
    payload = _metadata()
    payload["categories"][1]["name"] = "antarctic shag"  # type: ignore[index]
    specific = payload["annotations"][0]  # type: ignore[index]
    specific["genus"] = "leucocarbo"
    specific["species"] = "bransfieldensis"
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    birdnet_csv = tmp_path / "birdnet.csv"
    _write_birdnet_csv(
        birdnet_csv,
        ["BN16217,Leucocarbo atriceps,Imperial Shag,Aves,species,"],
    )

    plan = plan_big_bird(
        metadata_path,
        birdnet_csv_path=birdnet_csv,
        require_official_lock=False,
    )

    assert plan.crops[0].taxon_id == "birdnet:BN16217"
    assert plan.taxa[0].scientific_name == "Leucocarbo atriceps"

    _write_birdnet_csv(
        birdnet_csv,
        ["BN16217,Leucocarbo changed,Imperial Shag,Aves,species,"],
    )
    with pytest.raises(RuntimeError, match="override changed"):
        plan_big_bird(
            metadata_path,
            birdnet_csv_path=birdnet_csv,
            require_official_lock=False,
        )


def test_duplicate_ids_and_unsafe_paths_fail_before_download(tmp_path: Path) -> None:
    payload = _metadata()
    payload["annotations"].append(dict(payload["annotations"][0]))  # type: ignore[union-attr,index]
    metadata_path = tmp_path / "duplicate.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    birdnet_csv = tmp_path / "birdnet.csv"
    _write_birdnet_csv(birdnet_csv)

    with pytest.raises(RuntimeError, match="Duplicate Big Bird annotation id"):
        plan_big_bird(
            metadata_path,
            birdnet_csv_path=birdnet_csv,
            require_official_lock=False,
        )

    payload = _metadata()
    payload["images"][0]["file_name"] = "../escape.jpg"  # type: ignore[index]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe relative path"):
        plan_big_bird(
            metadata_path,
            birdnet_csv_path=birdnet_csv,
            require_official_lock=False,
        )


def test_strict_cub_audit_is_non_mutating_and_counts_labels_separately(tmp_path: Path) -> None:
    metadata_path, birdnet_csv, _source = _write_fixture(tmp_path)
    plan = plan_big_bird(
        metadata_path,
        birdnet_csv_path=birdnet_csv,
        require_official_lock=False,
    )

    retained = audit_big_bird_strict_cub(plan.crops, ["BN999"])
    excluded = audit_big_bird_strict_cub(plan.crops, ["BN1"])

    assert retained.retained_crops == retained.retained_taxa == 1
    assert excluded.excluded_crops == excluded.excluded_taxa == 1
    assert excluded.retained_crops == excluded.retained_taxa == 0
    assert len(plan.crops) == 1


def test_gcs_record_rejects_unsafe_names_and_invalid_md5() -> None:
    with pytest.raises(ValueError, match="unsafe relative path"):
        GCSObjectRecord("big-bird/../secret", base64.b64encode(b"x" * 16).decode(), 1, "1", "x")
    with pytest.raises(ValueError, match="valid base64"):
        GCSObjectRecord("big-bird/a.jpg", "not-base64", 1, "1", "image/jpeg")


def test_download_rejects_non_https_and_non_official_hosts(tmp_path: Path) -> None:
    for url in ("http://storage.googleapis.com/bucket/a.jpg", "https://example.com/a.jpg"):
        with pytest.raises(ValueError, match="unexpected URL"):
            _stream_download_to_path(
                url,
                tmp_path / "download",
                allowed_host="storage.googleapis.com",
            )


def test_metadata_only_download_is_streamed_locked_and_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b'{"official":"metadata"}'
    expected = hashlib.sha256(payload).hexdigest()
    calls = 0

    def fake_download(
        _url: str,
        destination: Path,
        **_kwargs: object,
    ) -> None:
        nonlocal calls
        calls += 1
        destination.write_bytes(payload)

    monkeypatch.setattr("ttvr.data.big_bird.BIG_BIRD_METADATA_OBSERVED_SHA256", expected)
    monkeypatch.setattr("ttvr.data.big_bird._stream_download_to_path", fake_download)

    first = download_big_bird_metadata(tmp_path / "dataset")
    second = download_big_bird_metadata(tmp_path / "dataset")

    assert first == second
    assert first.read_bytes() == payload
    assert calls == 1
    first.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Existing Big Bird metadata SHA-256 mismatch"):
        download_big_bird_metadata(tmp_path / "dataset")


def test_gcs_listing_requires_every_selected_objects_checksum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            {
                "items": [
                    {
                        "name": "big-bird/a.jpg",
                        "md5Hash": base64.b64encode(b"a" * 16).decode(),
                        "size": "10",
                        "generation": "1",
                        "contentType": "image/jpeg",
                    },
                    {"name": "big-bird/unselected.jpg"},
                ],
                "nextPageToken": "page-2",
            },
            {"items": []},
        ]
    )
    monkeypatch.setattr("ttvr.data.big_bird._request_json", lambda _url: next(responses))

    with pytest.raises(RuntimeError, match="missing 1 required objects"):
        _list_gcs_objects({"big-bird/a.jpg", "big-bird/b.jpg"})


def test_synthetic_metadata_is_rejected_by_default_official_lock(tmp_path: Path) -> None:
    metadata_path, birdnet_csv, _source = _write_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="metadata SHA-256 differs"):
        plan_big_bird(metadata_path, birdnet_csv_path=birdnet_csv)


def test_output_path_rejects_symlink_components(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    (root / "crops").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="contains a symlink"):
        _contained_output_path(root, "crops/annotation.png")
