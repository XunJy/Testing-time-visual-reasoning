from __future__ import annotations

import csv
import fcntl
import shutil
import stat
import subprocess
import zipfile
from dataclasses import fields
from pathlib import Path

import pytest

import ttvr.data.visual_wetlandbirds as wetlandbirds
from ttvr.data.bird_crops import md5_file
from ttvr.data.bird_manifest import BirdSample
from ttvr.data.visual_wetlandbirds import (
    VISUAL_WETLANDBIRDS_DATASET_ID,
    VISUAL_WETLANDBIRDS_FRAME_OFFSET,
    VISUAL_WETLANDBIRDS_FRAME_STRIDE,
    VISUAL_WETLANDBIRDS_TAXON_ALIGNMENTS,
    FFmpegFrameSyncPolicy,
    StrictCUBSourceAudit,
    VideoProbe,
    audit_annotation_counts,
    audit_strict_cub_exclusions,
    build_visual_wetlandbirds_taxa,
    iter_sampled_video_frames,
    probe_video,
    read_behavior_clip_count,
    read_behavior_ids,
    read_bounding_box_annotations,
    read_official_splits,
    read_species_ids,
    resolve_ffmpeg_frame_sync_policy,
    validate_video_archive,
    visual_wetlandbirds_group_id,
)


def _write_id_metadata(root: Path) -> tuple[Path, Path]:
    species = root / "species_ID.csv"
    species.write_text("id,species\n0,Test Bird\n1,Other Bird\n", encoding="utf-8")
    behaviors = root / "behaviors_ID.csv"
    behaviors.write_text("Activity,ID\nResting,0\nFlying,1\n", encoding="utf-8")
    return species, behaviors


def _write_annotations(path: Path) -> None:
    path.write_text(
        "species_id;species;video_name;frame;bounding_boxes\n"
        "0;Test Bird;001-test_bird;10;[(2, 2, 8, 7, 0, 0)]\n"
        "0;Test Bird;001-test_bird;0;[(1, 1, 4, 5, 0, 0), (7, 2, 9, 5, 1, 1)]\n"
        "0;Test Bird;001-test_bird;1;[(1, 1, 4, 5, 0, 0)]\n"
        "1;Other Bird;002-other_bird;2;[(3, 3, 6, 8, 1, 3)]\n",
        encoding="utf-8",
    )


def test_annotation_parser_and_paper_stride_are_per_video_frame_index(
    tmp_path: Path,
) -> None:
    species_path, behaviors_path = _write_id_metadata(tmp_path)
    annotations_path = tmp_path / "bounding_boxes.csv"
    _write_annotations(annotations_path)

    species = read_species_ids(species_path)
    behaviors = read_behavior_ids(behaviors_path)
    annotations, frame_rows = read_bounding_box_annotations(
        annotations_path,
        species_by_id=species,
        behavior_by_id=behaviors,
    )
    audit = audit_annotation_counts(annotations)

    assert frame_rows == 4
    assert len(annotations) == 5
    assert [(row.video_name, row.frame_index, row.bird_id) for row in annotations] == [
        ("001-test_bird", 0, 0),
        ("001-test_bird", 0, 1),
        ("001-test_bird", 1, 0),
        ("001-test_bird", 10, 0),
        ("002-other_bird", 2, 3),
    ]
    assert audit.stride == VISUAL_WETLANDBIRDS_FRAME_STRIDE == 10
    assert audit.offset == VISUAL_WETLANDBIRDS_FRAME_OFFSET == 0
    assert audit.stride_domain == "per_video_decoded_frame_index"
    assert audit.dense_bbox_count == 5
    assert audit.annotated_frame_rows == 4
    assert audit.sampled_annotated_frame_rows == 2
    assert audit.sampled_bbox_count == 3

    # Changing CSV row order cannot change which original frames are retained.
    reversed_audit = audit_annotation_counts(tuple(reversed(annotations)))
    assert reversed_audit == audit


def test_annotation_parser_fails_closed_on_duplicate_tracks_and_code_literals(
    tmp_path: Path,
) -> None:
    species_path, behaviors_path = _write_id_metadata(tmp_path)
    species = read_species_ids(species_path)
    behaviors = read_behavior_ids(behaviors_path)
    path = tmp_path / "bounding_boxes.csv"
    path.write_text(
        "species_id;species;video_name;frame;bounding_boxes\n"
        "0;Test Bird;001-test_bird;0;[(1, 1, 4, 5, 0, 0), (2, 2, 5, 6, 1, 0)]\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Duplicate bird id"):
        read_bounding_box_annotations(path, species_by_id=species, behavior_by_id=behaviors)

    path.write_text(
        "species_id;species;video_name;frame;bounding_boxes\n"
        "0;Test Bird;001-test_bird;0;__import__('os').system('false')\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Malformed bounding-box literal"):
        read_bounding_box_annotations(path, species_by_id=species, behavior_by_id=behaviors)


def test_publisher_unmapped_behavior_seven_is_preserved_but_new_ids_fail(
    tmp_path: Path,
) -> None:
    species_path, behaviors_path = _write_id_metadata(tmp_path)
    species = read_species_ids(species_path)
    behaviors = read_behavior_ids(behaviors_path)
    path = tmp_path / "bounding_boxes.csv"
    prefix = "species_id;species;video_name;frame;bounding_boxes\n"
    path.write_text(
        prefix + "0;Test Bird;001-test_bird;0;[(1, 1, 4, 5, 7, 0)]\n",
        encoding="utf-8",
    )

    rows, _count = read_bounding_box_annotations(
        path, species_by_id=species, behavior_by_id=behaviors
    )
    assert rows[0].behavior_id == 7

    path.write_text(
        prefix + "0;Test Bird;001-test_bird;0;[(1, 1, 4, 5, 8, 0)]\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Unknown behavior id"):
        read_bounding_box_annotations(path, species_by_id=species, behavior_by_id=behaviors)


def test_official_splits_are_video_disjoint_and_cover_annotations(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        '{"train_set":["001-a"],"val_set":["002-b"],"test_set":["003-c"]}',
        encoding="utf-8",
    )
    result = read_official_splits(split_path, expected_video_names=("001-a", "002-b", "003-c"))
    assert result == {"001-a": "train", "002-b": "validation", "003-c": "test"}

    split_path.write_text(
        '{"train_set":["001-a"],"val_set":["001-a"],"test_set":["003-c"]}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="multiple official splits"):
        read_official_splits(split_path, expected_video_names=("001-a", "002-b", "003-c"))


def test_behavior_clip_metadata_is_cross_checked(tmp_path: Path) -> None:
    path = tmp_path / "crops.csv"
    path.write_text(
        "video_name;bird_id;species_id;action_id;start_frame;end_frame\n"
        "001-a;0;2;1;0;9\n001-a;0;2;0;10;12\n",
        encoding="utf-8",
    )
    assert (
        read_behavior_clip_count(
            path, video_species_id={"001-a": 2}, behavior_by_id={0: "rest", 1: "fly"}
        )
        == 2
    )

    path.write_text(
        "video_name;bird_id;species_id;action_id;start_frame;end_frame\n001-a;0;3;1;0;9\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Invalid behavior clip"):
        read_behavior_clip_count(path, video_species_id={"001-a": 2}, behavior_by_id={1: "fly"})


def _write_video_zip(path: Path, names: tuple[str, ...]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("videos/", b"")
        for name in names:
            archive.writestr(f"videos/{name}.mp4", f"video:{name}".encode())


def test_video_archive_verification_checks_md5_members_and_safe_paths(tmp_path: Path) -> None:
    archive_path = tmp_path / "videos.zip"
    names = ("001-test_bird", "002-other_bird")
    _write_video_zip(archive_path, names)

    members = validate_video_archive(
        archive_path,
        expected_video_names=names,
        expected_md5=md5_file(archive_path),
    )
    assert [member.video_name for member in members] == list(names)
    assert all(member.file_size > 0 and len(member.crc32) == 8 for member in members)

    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        validate_video_archive(archive_path, expected_video_names=names, expected_md5="0" * 32)

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, mode="w") as archive:
        archive.writestr("../001-test_bird.mp4", b"unsafe")
    with pytest.raises(ValueError, match="unsafe relative path"):
        validate_video_archive(
            unsafe,
            expected_video_names=("001-test_bird",),
            expected_md5=md5_file(unsafe),
        )


def test_video_archive_rejects_symlink_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("videos/001-test_bird.mp4")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(info, b"/etc/passwd")

    with pytest.raises(RuntimeError, match="Non-regular"):
        validate_video_archive(
            archive_path,
            expected_video_names=("001-test_bird",),
            expected_md5=md5_file(archive_path),
        )


def test_metadata_downloader_can_never_fetch_the_full_video_archive(
    tmp_path: Path,
) -> None:
    for url in (
        "https://zenodo.org/api/records/15696105/files/videos.zip/content",
        "https://example.com/api/records/15696105/files/species_ID.csv/content",
        "http://zenodo.org/api/records/15696105/files/species_ID.csv/content",
    ):
        with pytest.raises(ValueError, match="Refusing unapproved"):
            wetlandbirds._download_metadata_file(url, tmp_path / "download")


def test_preparation_lock_fails_fast_under_concurrent_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        wetlandbirds,
        "_prepare_visual_wetlandbirds_locked",
        lambda *_args, **_kwargs: "prepared",
    )
    assert (
        wetlandbirds.prepare_visual_wetlandbirds(tmp_path, birdnet_csv_path=tmp_path / "unused.csv")
        == "prepared"
    )

    lock_path = tmp_path / ".visual-wetlandbirds.prepare.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another Visual WetlandBirds preparation"):
            wetlandbirds.prepare_visual_wetlandbirds(
                tmp_path, birdnet_csv_path=tmp_path / "unused.csv"
            )


def _write_reviewed_birdnet_csv(path: Path, *, change_first: bool = False) -> None:
    columns = [
        "birdnet_id",
        "scientific_name",
        "common_name",
        "taxon_group",
        "record_type",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, alignment in enumerate(VISUAL_WETLANDBIRDS_TAXON_ALIGNMENTS):
            writer.writerow(
                {
                    "birdnet_id": alignment.birdnet_id,
                    "scientific_name": (
                        "Changed bird"
                        if change_first and index == 0
                        else alignment.accepted_scientific_name
                    ),
                    "common_name": alignment.accepted_common_name,
                    "taxon_group": "Aves",
                    "record_type": "species",
                }
            )


def test_all_specific_taxa_are_locked_to_birdnet_without_target_filter(
    tmp_path: Path,
) -> None:
    birdnet = tmp_path / "birdnet.csv"
    _write_reviewed_birdnet_csv(birdnet)
    source_species = {
        row.source_species_id: row.source_common_name
        for row in VISUAL_WETLANDBIRDS_TAXON_ALIGNMENTS
    }

    taxa, by_source = build_visual_wetlandbirds_taxa(source_species, birdnet)

    assert len(taxa) == len(by_source) == 13
    assert all(taxon.taxon_id.startswith("birdnet:BN") for taxon in taxa)
    assert by_source[9].common_name == "Common Moorhen"
    # Mallard and Gadwall stay in the canonical source taxonomy.
    assert by_source[11].taxon_id == "birdnet:BN08405"
    assert by_source[12].taxon_id == "birdnet:BN00713"

    _write_reviewed_birdnet_csv(birdnet, change_first=True)
    with pytest.raises(RuntimeError, match="identity changed"):
        build_visual_wetlandbirds_taxa(source_species, birdnet)


def _sample(sample_id: str, taxon_id: str, split: str = "train") -> BirdSample:
    return BirdSample(
        dataset_id=VISUAL_WETLANDBIRDS_DATASET_ID,
        source_sample_id=sample_id,
        source_split=split,
        relative_path=f"crops/{sample_id}.png",
        image_uri=f"https://zenodo.org/example#{sample_id}",
        group_id=f"group:{sample_id}",
        raw_label=taxon_id,
        taxon_id=taxon_id,
        sha256="0" * 64,
        phash="0" * 16,
        license="CC BY 4.0",
        author="Dataset authors",
        source="Visual WetlandBirds",
    )


def test_strict_cub_audit_is_separate_from_canonical_samples() -> None:
    samples = (
        _sample("mallard", "birdnet:BN00713"),
        _sample("gadwall", "birdnet:BN08405", "validation"),
        _sample("coot", "birdnet:BN05715"),
        _sample("ibis", "birdnet:BN11703", "test"),
    )

    audit = audit_strict_cub_exclusions(samples, ("birdnet:BN00713", "birdnet:BN08405"))

    assert audit.source_sample_count == 4
    assert audit.retained_sample_count == 2
    assert audit.retained_taxon_count == 2
    assert audit.excluded_counts_by_taxon == {
        "birdnet:BN00713": 1,
        "birdnet:BN08405": 1,
    }
    assert len(samples) == 4  # audit is non-mutating


def test_strict_cub_audit_schema_has_no_duplicate_or_implicit_fields() -> None:
    assert [field.name for field in fields(StrictCUBSourceAudit)] == [
        "source_sample_count",
        "source_taxon_count",
        "retained_sample_count",
        "retained_taxon_count",
        "excluded_sample_count",
        "excluded_taxon_ids",
        "excluded_counts_by_taxon",
        "retained_counts_by_split",
    ]


def test_group_id_preserves_original_video_and_track_identity() -> None:
    first = visual_wetlandbirds_group_id("001-test_bird", 7)
    same_track_later_frame = visual_wetlandbirds_group_id("001-test_bird", 7)

    assert first == same_track_later_frame
    assert first != visual_wetlandbirds_group_id("001-test_bird", 8)
    assert first != visual_wetlandbirds_group_id("002-test_bird", 7)
    with pytest.raises(ValueError):
        visual_wetlandbirds_group_id("../video", 0)


def test_ffmpeg_frame_sync_capability_prefers_modern_stream_option() -> None:
    help_text = (
        "-vsync <> set video sync method globally; deprecated, use -fps_mode\n"
        "-fps_mode[:<stream_spec>] set framerate mode for matching video streams\n"
    )

    policy = wetlandbirds._select_ffmpeg_frame_sync_policy(help_text)

    assert policy == FFmpegFrameSyncPolicy(cli_option="-fps_mode", cli_value="passthrough")
    assert policy.command_arguments() == ("-fps_mode", "passthrough")
    assert policy.to_dict() == {
        "cli_option": "-fps_mode",
        "cli_value": "passthrough",
        "selection_basis": "ffmpeg_hide_banner_help_full",
        "semantics": "passthrough_one_output_frame_per_selected_input_frame",
    }


@pytest.mark.parametrize(
    "option",
    (
        "-fps_mode[:<stream_spec>]",
        "-fps_mode[:stream_specifier]",
        "-fps_mode",
    ),
)
def test_ffmpeg_frame_sync_capability_accepts_documented_stream_spec_spellings(
    option: str,
) -> None:
    policy = wetlandbirds._select_ffmpeg_frame_sync_policy(
        f"{option} set framerate mode for matching video streams\n"
    )

    assert policy.command_arguments() == ("-fps_mode", "passthrough")


def test_ffmpeg_frame_sync_capability_uses_audited_ffmpeg_44_alias() -> None:
    help_text = (
        "ffmpeg version 4.4.2\n"
        "-vsync              video sync method\n"
        "The description may mention -fps_mode without advertising that option.\n"
    )

    policy = wetlandbirds._select_ffmpeg_frame_sync_policy(help_text)

    assert policy.command_arguments() == ("-vsync", "0")
    assert policy.semantics == "passthrough_one_output_frame_per_selected_input_frame"


def test_ffmpeg_frame_sync_capability_fails_closed_without_passthrough_option() -> None:
    with pytest.raises(RuntimeError, match="neither -fps_mode nor -vsync"):
        wetlandbirds._select_ffmpeg_frame_sync_policy("-pix_fmt format")


def test_ffmpeg_frame_sync_resolver_uses_explicit_capability_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 30,
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="-vsync parameter set video sync method globally\n",
            stderr="",
        )

    monkeypatch.setattr(wetlandbirds.subprocess, "run", fake_run)

    policy = resolve_ffmpeg_frame_sync_policy("ffmpeg-4.4")

    assert observed == [["ffmpeg-4.4", "-hide_banner", "-h", "full"]]
    assert policy.command_arguments() == ("-vsync", "0")


def test_frame_extraction_does_not_retry_unrelated_decoder_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []

    def fail_to_launch(command: list[str], **_kwargs: object) -> None:
        observed.append(command)
        raise OSError("synthetic decoder failure")

    monkeypatch.setattr(wetlandbirds.subprocess, "Popen", fail_to_launch)
    frames = iter_sampled_video_frames(
        "broken.mp4",
        VideoProbe(width=32, height=24, frame_count=21, codec_name="mpeg4"),
        ffmpeg_binary="ffmpeg-4.4",
        frame_sync_policy=FFmpegFrameSyncPolicy(cli_option="-vsync", cli_value="0"),
    )

    with pytest.raises(RuntimeError, match="Could not launch ffmpeg"):
        next(frames)

    assert len(observed) == 1
    assert "-vsync" in observed[0]
    assert "-fps_mode" not in observed[0]


def test_cpu_video_decoder_selects_frames_zero_ten_twenty(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg/ffprobe are unavailable")
    video = tmp_path / "tiny.mp4"
    command = [
        ffmpeg,
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=32x24:r=10:d=2.1",
        "-frames:v",
        "21",
        "-c:v",
        "mpeg4",
        "-y",
        str(video),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        pytest.skip(f"local ffmpeg codec is unavailable: {error}")

    probe = probe_video(video, ffprobe_binary=ffprobe)
    frames = list(iter_sampled_video_frames(video, probe, ffmpeg_binary=ffmpeg))

    assert probe == VideoProbe(width=32, height=24, frame_count=21, codec_name="mpeg4")
    assert [frame_index for frame_index, _frame in frames] == [0, 10, 20]
    assert all(frame.size == (32, 24) for _index, frame in frames)
    assert all(frame.mode == "RGB" for _index, frame in frames)
    assert frames[0][1].tobytes() == frames[1][1].tobytes()
