"""Audited Visual WetlandBirds v4 frame-crop training source.

The publisher distributes 178 annotated videos rather than independent
images.  This module derives a reproducible image source while retaining the
correlation that exists inside a video and bird track:

* the official ``videos.zip`` and annotation checksums are verified;
* frame ``0`` from each consecutive group of ten frames is selected, matching
  the species-classification downsampling described in the paper;
* every annotated bird in a selected frame is converted to the shared square
  context crop recipe; and
* the original video, frame, track, bounding box, and crop geometry are kept
  in immutable provenance manifests.

Target-dataset exclusions deliberately do not belong here.  In particular,
Mallard and Gadwall remain in the canonical source manifest; an experiment
runner must apply its locked CUB crosswalk later.
"""

from __future__ import annotations

import ast
import csv
import fcntl
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from PIL import Image
from PIL import __version__ as PILLOW_VERSION

from .bird_crops import (
    DEFAULT_CONTEXT_SCALE,
    SQUARE_CONTEXT_RECIPE_ID,
    clip_bbox_xyxy_to_image,
    md5_file,
    save_crop_png_atomic,
    square_context_crop,
    validate_relative_posix_path,
)
from .bird_manifest import BirdSample, BirdTaxon, sha256_file, write_json, write_jsonl
from .birdnet import (
    BIRDNET_TAXONOMY_AUTHORITY,
    BIRDNET_TAXONOMY_VERSION,
)

VISUAL_WETLANDBIRDS_DATASET_ID = "visual-wetlandbirds-v4-stride10-crops"
VISUAL_WETLANDBIRDS_VERSION = "v4"
VISUAL_WETLANDBIRDS_RECORD_ID = "15696105"
VISUAL_WETLANDBIRDS_DOI = "10.5281/zenodo.15696105"
VISUAL_WETLANDBIRDS_SOURCE_URL = "https://zenodo.org/records/15696105"
VISUAL_WETLANDBIRDS_API_ROOT = "https://zenodo.org/api/records/15696105/files"
VISUAL_WETLANDBIRDS_VIDEO_URL = f"{VISUAL_WETLANDBIRDS_API_ROOT}/videos.zip/content"
VISUAL_WETLANDBIRDS_VIDEO_MD5 = "95eae83aceb2d12803f5f567c31060a9"
VISUAL_WETLANDBIRDS_LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
VISUAL_WETLANDBIRDS_AUTHORS = (
    "Javier Rodriguez-Juan; David Ortiz-Perez; Manuel Benavent-Lledo; "
    "David Mulero-Pérez; Pablo Ruiz-Ponce; Adrian Orihuela-Torres; "
    "Jose Garcia-Rodriguez; Esther Sebastián-González"
)

VISUAL_WETLANDBIRDS_FRAME_STRIDE = 10
VISUAL_WETLANDBIRDS_FRAME_OFFSET = 0
VISUAL_WETLANDBIRDS_CONTEXT_SCALE = DEFAULT_CONTEXT_SCALE
# The checksum-locked v4 bounding-box CSV uses behavior id 7 for 15,012 boxes,
# while behaviors_ID.csv defines only ids 0--6.  Species training does not use
# this field.  We preserve the publisher value and surface the discrepancy in
# source.json instead of silently relabelling it.
VISUAL_WETLANDBIRDS_UNMAPPED_BEHAVIOR_IDS = (7,)
VISUAL_WETLANDBIRDS_CROP_PROTOCOL = (
    f"paper-species-downsample-first-frame-stride10+{SQUARE_CONTEXT_RECIPE_ID}"
)

VISUAL_WETLANDBIRDS_METADATA_MD5: Mapping[str, str] = {
    "behaviors_ID.csv": "6fdfc078d37c66aed404198af3499928",
    "bounding_boxes.csv": "630536063ab5268d16ea8bce0366e3f4",
    "crops.csv": "ecafc3987aad8ec016ccb7f907ab5e11",
    "species_ID.csv": "aa9fc2ea8a5a3e3298621c1b5d503d36",
    "splits.json": "3d28c46699b20359d9e37541f7a3c71d",
}

_EXPECTED_VIDEO_COUNT = 178
_EXPECTED_TAXON_COUNT = 13
_EXPECTED_ANNOTATED_FRAME_ROWS = 120_694
_EXPECTED_DENSE_BOX_COUNT = 156_328
_EXPECTED_BEHAVIOR_CLIP_COUNT = 1_469
_EXPECTED_SAMPLED_CROP_COUNT = 15_725
_VIDEO_NAME_RE = re.compile(r"^[0-9]{3}-[a-z0-9_]+$")
_ZENODO_HOST = "zenodo.org"


@dataclass(frozen=True, slots=True)
class WetlandTaxonAlignment:
    """One exact source-label alignment to the locked BirdNET taxonomy."""

    source_species_id: int
    source_common_name: str
    source_scientific_name: str
    birdnet_id: str
    accepted_scientific_name: str
    accepted_common_name: str

    @property
    def taxon_id(self) -> str:
        return f"birdnet:{self.birdnet_id}"


# Scientific source names are taken from Table 3 of the associated paper.
# Accepted names/ids are the exact AviList rows in BirdNET+ v0.3-Jul2026.
VISUAL_WETLANDBIRDS_TAXON_ALIGNMENTS: tuple[WetlandTaxonAlignment, ...] = (
    WetlandTaxonAlignment(
        0, "White Wagtail", "Motacilla alba", "BN09124", "Motacilla alba", "White Wagtail"
    ),
    WetlandTaxonAlignment(
        1, "Glossy Ibis", "Plegadis falcinellus", "BN11703", "Plegadis falcinellus", "Glossy Ibis"
    ),
    WetlandTaxonAlignment(
        2, "Squacco Heron", "Ardeola ralloides", "BN01213", "Ardeola ralloides", "Squacco Heron"
    ),
    WetlandTaxonAlignment(
        3,
        "Black-winged Stilt",
        "Himantopus himantopus",
        "BN06642",
        "Himantopus himantopus",
        "Black-winged Stilt",
    ),
    WetlandTaxonAlignment(
        4,
        "Yellow-legged Gull",
        "Larus michahellis",
        "BN07555",
        "Larus michahellis",
        "Yellow-legged Gull",
    ),
    WetlandTaxonAlignment(
        5,
        "Northern Shoveler",
        "Spatula clypeata",
        "BN14030",
        "Spatula clypeata",
        "Northern Shoveler",
    ),
    WetlandTaxonAlignment(
        6,
        "Black-headed Gull",
        "Chroicocephalus ridibundus",
        "BN03028",
        "Chroicocephalus ridibundus",
        "Black-headed Gull",
    ),
    WetlandTaxonAlignment(
        7, "Eurasian Coot", "Fulica atra", "BN05715", "Fulica atra", "Eurasian Coot"
    ),
    WetlandTaxonAlignment(
        8,
        "Little Ringed Plover",
        "Charadrius dubius",
        "BN14952",
        "Thinornis dubius",
        "Little Ringed Plover",
    ),
    WetlandTaxonAlignment(
        9,
        "Eurasian Moorhen",
        "Gallinula chloropus",
        "BN05803",
        "Gallinula chloropus",
        "Common Moorhen",
    ),
    WetlandTaxonAlignment(
        10, "Eurasian Magpie", "Pica pica", "BN11450", "Pica pica", "Eurasian Magpie"
    ),
    WetlandTaxonAlignment(
        11, "Gadwall", "Mareca strepera", "BN08405", "Mareca strepera", "Gadwall"
    ),
    WetlandTaxonAlignment(
        12, "Mallard", "Anas platyrhynchos", "BN00713", "Anas platyrhynchos", "Mallard"
    ),
)


@dataclass(frozen=True, slots=True)
class WetlandAnnotation:
    species_id: int
    species: str
    video_name: str
    frame_index: int
    bbox_xyxy: tuple[float, float, float, float]
    behavior_id: int
    bird_id: int


@dataclass(frozen=True, slots=True)
class VideoArchiveMember:
    video_name: str
    member_name: str
    file_size: int
    compressed_size: int
    crc32: str


@dataclass(frozen=True, slots=True)
class VideoProbe:
    width: int
    height: int
    frame_count: int
    codec_name: str


@dataclass(frozen=True, slots=True)
class FFmpegFrameSyncPolicy:
    """One explicitly advertised ffmpeg passthrough-sync spelling."""

    cli_option: str
    cli_value: str
    selection_basis: str = "ffmpeg_hide_banner_help_full"
    semantics: str = "passthrough_one_output_frame_per_selected_input_frame"

    def __post_init__(self) -> None:
        if (self.cli_option, self.cli_value) not in {
            ("-fps_mode", "passthrough"),
            ("-vsync", "0"),
        }:
            raise ValueError("Unsupported ffmpeg passthrough frame-sync policy")

    def command_arguments(self) -> tuple[str, str]:
        return self.cli_option, self.cli_value

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VisualWetlandBirdsPreparation:
    video_archive_md5: str
    metadata_md5: dict[str, str]
    birdnet_csv_sha256: str
    video_count: int
    taxon_count: int
    behavior_clip_count: int
    annotated_frame_rows: int
    dense_bbox_count: int
    sampled_annotated_frame_rows: int
    sampled_crop_count: int
    decoded_frame_count: int
    decoded_sampled_frame_count: int
    split_video_counts: dict[str, int]
    split_annotated_frame_counts: dict[str, int]
    split_dense_bbox_counts: dict[str, int]
    split_crop_counts: dict[str, int]
    behavior_id_counts: dict[str, int]
    unmapped_behavior_box_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WetlandAnnotationCountAudit:
    """Pure count audit over expanded publisher annotations."""

    stride: int
    offset: int
    stride_domain: str
    video_count: int
    taxon_count: int
    annotated_frame_rows: int
    dense_bbox_count: int
    sampled_annotated_frame_rows: int
    sampled_bbox_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StrictCUBSourceAudit:
    """A target-filter audit produced without mutating the source manifest."""

    source_sample_count: int
    source_taxon_count: int
    retained_sample_count: int
    retained_taxon_count: int
    excluded_sample_count: int
    excluded_taxon_ids: tuple[str, ...]
    excluded_counts_by_taxon: dict[str, int]
    retained_counts_by_split: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressCallback = Callable[[dict[str, Any]], None]


def audit_strict_cub_exclusions(
    samples: Sequence[BirdSample],
    excluded_taxon_ids: Iterable[str],
) -> StrictCUBSourceAudit:
    """Report the effect of a runner-provided strict CUB exclusion set.

    The caller should construct ``excluded_taxon_ids`` from the locked CUB to
    BirdNET crosswalk.  Keeping it as an argument prevents target knowledge
    from leaking into this canonical source preparation.
    """

    if not samples:
        raise ValueError("samples must not be empty")
    excluded = set(excluded_taxon_ids)
    retained = [sample for sample in samples if sample.taxon_id not in excluded]
    removed = [sample for sample in samples if sample.taxon_id in excluded]
    removed_counts = Counter(sample.taxon_id for sample in removed)
    retained_splits = Counter(sample.source_split for sample in retained)
    return StrictCUBSourceAudit(
        source_sample_count=len(samples),
        source_taxon_count=len({sample.taxon_id for sample in samples}),
        retained_sample_count=len(retained),
        retained_taxon_count=len({sample.taxon_id for sample in retained}),
        excluded_sample_count=len(removed),
        excluded_taxon_ids=tuple(sorted(removed_counts)),
        excluded_counts_by_taxon=dict(sorted(removed_counts.items())),
        retained_counts_by_split=dict(sorted(retained_splits.items())),
    )


def audit_annotation_counts(
    annotations: Sequence[WetlandAnnotation],
    *,
    stride: int = VISUAL_WETLANDBIRDS_FRAME_STRIDE,
    offset: int = VISUAL_WETLANDBIRDS_FRAME_OFFSET,
) -> WetlandAnnotationCountAudit:
    """Count dense/kept boxes using per-video decoded frame indices.

    This function needs no videos and makes the sampling convention directly
    testable.  It never strides over global CSV rows or over bird tracks.
    """

    if not annotations:
        raise ValueError("annotations must not be empty")
    if stride <= 0 or not 0 <= offset < stride:
        raise ValueError("stride must be positive and offset must be within one stride")
    sampled = [item for item in annotations if item.frame_index % stride == offset]
    return WetlandAnnotationCountAudit(
        stride=stride,
        offset=offset,
        stride_domain="per_video_decoded_frame_index",
        video_count=len({item.video_name for item in annotations}),
        taxon_count=len({item.species_id for item in annotations}),
        annotated_frame_rows=len({(item.video_name, item.frame_index) for item in annotations}),
        dense_bbox_count=len(annotations),
        sampled_annotated_frame_rows=len({(item.video_name, item.frame_index) for item in sampled}),
        sampled_bbox_count=len(sampled),
    )


def _notify(progress: ProgressCallback | None, **value: Any) -> None:
    if progress is not None:
        progress(value)


def _metadata_url(file_name: str) -> str:
    if file_name not in VISUAL_WETLANDBIRDS_METADATA_MD5:
        raise ValueError(f"Not an approved metadata file: {file_name}")
    return f"{VISUAL_WETLANDBIRDS_API_ROOT}/{file_name}/content"


def _download_metadata_file(url: str, destination: Path) -> None:
    """Download one small approved Zenodo metadata object, never videos.zip."""

    parsed = urlsplit(url)
    expected_prefix = f"/api/records/{VISUAL_WETLANDBIRDS_RECORD_ID}/files/"
    if (
        parsed.scheme != "https"
        or parsed.hostname != _ZENODO_HOST
        or not parsed.path.startswith(expected_prefix)
        or parsed.path.endswith("/videos.zip/content")
    ):
        raise ValueError(f"Refusing unapproved Visual WetlandBirds URL: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite metadata: {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "ttvr-birdmix/1.0"})
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.part-", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        total = 0
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > 32 * 1024 * 1024:
                    raise RuntimeError("Refusing unexpectedly large metadata download")
                out.write(chunk)
        # A hard-link creation is atomic and fails if another preparer won the
        # race.  Unlike Path.replace(), it can never overwrite publisher data.
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"Concurrent metadata preparation created: {destination}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def ensure_official_metadata(
    root: Path | str,
    *,
    download_missing: bool = True,
) -> dict[str, str]:
    """Ensure and verify all small v4 metadata files without downloading video."""

    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    actual: dict[str, str] = {}
    for file_name, expected in VISUAL_WETLANDBIRDS_METADATA_MD5.items():
        path = root_path / file_name
        if not path.is_file():
            if not download_missing:
                raise FileNotFoundError(f"Missing Visual WetlandBirds metadata: {path}")
            _download_metadata_file(_metadata_url(file_name), path)
        digest = md5_file(path)
        if digest != expected:
            raise RuntimeError(
                f"Visual WetlandBirds {file_name} MD5 mismatch: expected {expected}, found {digest}"
            )
        actual[file_name] = digest
    return dict(sorted(actual.items()))


def _read_id_csv(path: Path, *, key_column: str, value_column: str) -> dict[int, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != {key_column, value_column}:
            raise RuntimeError(f"Unexpected columns in {path.name}: {reader.fieldnames}")
        result: dict[int, str] = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                identifier = int(row[key_column])
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"Invalid id at {path}:{line_number}") from error
            name = row[value_column].strip()
            if identifier < 0 or not name or identifier in result:
                raise RuntimeError(f"Invalid or duplicate id at {path}:{line_number}")
            result[identifier] = name
    if not result:
        raise RuntimeError(f"Empty id mapping: {path}")
    return result


def read_species_ids(path: Path | str) -> dict[int, str]:
    return _read_id_csv(Path(path), key_column="id", value_column="species")


def read_behavior_ids(path: Path | str) -> dict[int, str]:
    return _read_id_csv(Path(path), key_column="ID", value_column="Activity")


def _integral(value: Any, *, label: str, location: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be an integer at {location}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} must be numeric at {location}") from error
    if not math.isfinite(number) or not number.is_integer():
        raise RuntimeError(f"{label} must be an integer at {location}")
    return int(number)


def _parse_bbox_tuple(
    value: Any, *, location: str
) -> tuple[tuple[float, float, float, float], int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 6:
        raise RuntimeError(f"Bounding box must contain six values at {location}")
    try:
        coordinates = tuple(float(item) for item in value[:4])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Bounding box coordinates must be numeric at {location}") from error
    if not all(math.isfinite(item) for item in coordinates):
        raise RuntimeError(f"Bounding box coordinates must be finite at {location}")
    x_min, y_min, x_max, y_max = coordinates
    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError(f"Bounding box must have positive area at {location}")
    behavior_id = _integral(value[4], label="behavior_id", location=location)
    bird_id = _integral(value[5], label="bird_id", location=location)
    if behavior_id < 0 or bird_id < 0:
        raise RuntimeError(f"Behavior and bird ids must be non-negative at {location}")
    return (x_min, y_min, x_max, y_max), behavior_id, bird_id


def read_bounding_box_annotations(
    path: Path | str,
    *,
    species_by_id: Mapping[int, str],
    behavior_by_id: Mapping[int, str],
) -> tuple[tuple[WetlandAnnotation, ...], int]:
    """Expand every official frame row into one record per annotated bird."""

    source = Path(path)
    expected_columns = {
        "species_id",
        "species",
        "video_name",
        "frame",
        "bounding_boxes",
    }
    annotations: list[WetlandAnnotation] = []
    seen_frame_rows: set[tuple[str, int]] = set()
    seen_tracks: set[tuple[str, int, int]] = set()
    video_species: dict[str, tuple[int, str]] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if set(reader.fieldnames or ()) != expected_columns:
            raise RuntimeError(f"Unexpected columns in {source.name}: {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            location = f"{source}:{line_number}"
            species_id = _integral(row["species_id"], label="species_id", location=location)
            species = row["species"].strip()
            if species_by_id.get(species_id) != species:
                raise RuntimeError(f"Species id/name mismatch at {location}")
            video_name = row["video_name"].strip()
            if not _VIDEO_NAME_RE.fullmatch(video_name):
                raise RuntimeError(f"Unsafe video name at {location}: {video_name!r}")
            identity = (species_id, species)
            previous = video_species.setdefault(video_name, identity)
            if previous != identity:
                raise RuntimeError(f"Video changes species at {location}: {video_name}")
            frame_index = _integral(row["frame"], label="frame", location=location)
            if frame_index < 0 or (video_name, frame_index) in seen_frame_rows:
                raise RuntimeError(f"Invalid or duplicate frame row at {location}")
            seen_frame_rows.add((video_name, frame_index))
            try:
                raw_boxes = ast.literal_eval(row["bounding_boxes"])
            except (SyntaxError, ValueError) as error:
                raise RuntimeError(f"Malformed bounding-box literal at {location}") from error
            if not isinstance(raw_boxes, list) or not raw_boxes:
                raise RuntimeError(f"Bounding-box list must not be empty at {location}")
            for raw_box in raw_boxes:
                bbox, behavior_id, bird_id = _parse_bbox_tuple(raw_box, location=location)
                if behavior_id not in behavior_by_id and (
                    behavior_id not in VISUAL_WETLANDBIRDS_UNMAPPED_BEHAVIOR_IDS
                ):
                    raise RuntimeError(f"Unknown behavior id at {location}: {behavior_id}")
                track_key = (video_name, frame_index, bird_id)
                if track_key in seen_tracks:
                    raise RuntimeError(f"Duplicate bird id in one frame at {location}: {bird_id}")
                seen_tracks.add(track_key)
                annotations.append(
                    WetlandAnnotation(
                        species_id=species_id,
                        species=species,
                        video_name=video_name,
                        frame_index=frame_index,
                        bbox_xyxy=bbox,
                        behavior_id=behavior_id,
                        bird_id=bird_id,
                    )
                )
    if not annotations:
        raise RuntimeError(f"No bounding-box annotations found in {source}")
    annotations.sort(key=lambda item: (item.video_name, item.frame_index, item.bird_id))
    return tuple(annotations), len(seen_frame_rows)


def read_official_splits(
    path: Path | str,
    *,
    expected_video_names: Iterable[str],
) -> dict[str, str]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    key_to_split = {"train_set": "train", "val_set": "validation", "test_set": "test"}
    if not isinstance(payload, dict) or set(payload) != set(key_to_split):
        raise RuntimeError("Visual WetlandBirds splits.json has unexpected keys")
    result: dict[str, str] = {}
    for key, split in key_to_split.items():
        names = payload[key]
        if not isinstance(names, list) or not names:
            raise RuntimeError(f"Visual WetlandBirds {key} must be a non-empty list")
        for value in names:
            if not isinstance(value, str) or not _VIDEO_NAME_RE.fullmatch(value):
                raise RuntimeError(f"Unsafe video name in {key}: {value!r}")
            if value in result:
                raise RuntimeError(f"Video occurs in multiple official splits: {value}")
            result[value] = split
    expected = set(expected_video_names)
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise RuntimeError(
            f"Official split/video mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    return result


def read_behavior_clip_count(
    path: Path | str,
    *,
    video_species_id: Mapping[str, int],
    behavior_by_id: Mapping[int, str],
) -> int:
    """Validate the official behavior-clip metadata used by the publication."""

    source = Path(path)
    required = {
        "video_name",
        "bird_id",
        "species_id",
        "action_id",
        "start_frame",
        "end_frame",
    }
    count = 0
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if set(reader.fieldnames or ()) != required:
            raise RuntimeError(f"Unexpected columns in {source.name}: {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            location = f"{source}:{line_number}"
            video = row["video_name"].strip()
            species_id = _integral(row["species_id"], label="species_id", location=location)
            behavior_id = _integral(row["action_id"], label="action_id", location=location)
            bird_id = _integral(row["bird_id"], label="bird_id", location=location)
            start = _integral(row["start_frame"], label="start_frame", location=location)
            end = _integral(row["end_frame"], label="end_frame", location=location)
            if (
                video_species_id.get(video) != species_id
                or behavior_id not in behavior_by_id
                or bird_id < 0
                or start < 0
                or end < start
            ):
                raise RuntimeError(f"Invalid behavior clip at {location}")
            count += 1
    if count == 0:
        raise RuntimeError("Behavior clip metadata is empty")
    return count


def build_visual_wetlandbirds_taxa(
    species_by_id: Mapping[int, str],
    birdnet_csv_path: Path | str,
) -> tuple[tuple[BirdTaxon, ...], dict[int, BirdTaxon]]:
    """Validate all 13 reviewed mappings against the runtime BirdNET CSV."""

    alignments = {row.source_species_id: row for row in VISUAL_WETLANDBIRDS_TAXON_ALIGNMENTS}
    expected_species = {
        identifier: row.source_common_name for identifier, row in alignments.items()
    }
    if dict(species_by_id) != expected_species:
        raise RuntimeError("Visual WetlandBirds species_ID.csv changed from reviewed v4 taxonomy")
    csv_path = Path(birdnet_csv_path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing BirdNET taxonomy CSV: {csv_path}")
    rows_by_id: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "birdnet_id",
            "scientific_name",
            "common_name",
            "taxon_group",
            "record_type",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"BirdNET CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if row.get("taxon_group") != "Aves" or row.get("record_type") != "species":
                continue
            identifier = row["birdnet_id"]
            if identifier in rows_by_id:
                raise RuntimeError(f"Duplicate BirdNET species id: {identifier}")
            rows_by_id[identifier] = row
    by_source: dict[int, BirdTaxon] = {}
    for identifier, alignment in sorted(alignments.items()):
        row = rows_by_id.get(alignment.birdnet_id)
        if row is None:
            raise RuntimeError(f"Reviewed BirdNET id is missing: {alignment.birdnet_id}")
        identity = (row["scientific_name"], row["common_name"])
        expected = (alignment.accepted_scientific_name, alignment.accepted_common_name)
        if identity != expected:
            raise RuntimeError(
                f"Reviewed BirdNET identity changed for {alignment.birdnet_id}: "
                f"expected {expected!r}, found {identity!r}"
            )
        by_source[identifier] = BirdTaxon(
            taxon_id=alignment.taxon_id,
            scientific_name=alignment.accepted_scientific_name,
            common_name=alignment.accepted_common_name,
            taxonomy_source=BIRDNET_TAXONOMY_AUTHORITY,
            taxonomy_version=BIRDNET_TAXONOMY_VERSION,
        )
    taxa = tuple(sorted(by_source.values(), key=lambda item: item.taxon_id))
    return taxa, by_source


def _safe_zip_member(info: zipfile.ZipInfo) -> str:
    raw_name = info.filename
    if info.is_dir():
        if not raw_name.endswith("/"):
            raise RuntimeError(f"Malformed archive directory entry: {raw_name!r}")
        validate_relative_posix_path(raw_name[:-1])
        name = raw_name
    else:
        name = validate_relative_posix_path(raw_name)
    if info.flag_bits & 0x1:
        raise RuntimeError(f"Encrypted archive member is not allowed: {name}")
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if info.is_dir() and file_type not in {0, stat.S_IFDIR}:
        raise RuntimeError(f"Invalid archive directory type: {name}")
    if not info.is_dir() and file_type not in {0, stat.S_IFREG}:
        raise RuntimeError(f"Non-regular archive member is not allowed: {name}")
    return name


def validate_video_archive(
    archive_path: Path | str,
    *,
    expected_video_names: Iterable[str],
    expected_md5: str = VISUAL_WETLANDBIRDS_VIDEO_MD5,
) -> tuple[VideoArchiveMember, ...]:
    """Verify publisher checksum, safe names, and the exact video member set."""

    source = Path(archive_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Missing Visual WetlandBirds video archive: {source}")
    actual_md5 = md5_file(source)
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"Visual WetlandBirds videos.zip MD5 mismatch: "
            f"expected {expected_md5}, found {actual_md5}"
        )
    expected_members = {f"videos/{name}.mp4" for name in expected_video_names}
    members: list[VideoArchiveMember] = []
    seen: set[str] = set()
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            name = _safe_zip_member(info)
            if info.is_dir():
                if name != "videos/":
                    raise RuntimeError(f"Unexpected directory in videos.zip: {name}")
                continue
            if name in seen:
                raise RuntimeError(f"Duplicate member in videos.zip: {name}")
            seen.add(name)
            if name not in expected_members:
                raise RuntimeError(f"Unexpected member in videos.zip: {name}")
            video_name = Path(name).stem
            if info.file_size <= 0 or info.compress_size <= 0:
                raise RuntimeError(f"Empty video archive member: {name}")
            members.append(
                VideoArchiveMember(
                    video_name=video_name,
                    member_name=name,
                    file_size=info.file_size,
                    compressed_size=info.compress_size,
                    crc32=f"{info.CRC:08x}",
                )
            )
    if seen != expected_members:
        missing = sorted(expected_members - seen)
        raise RuntimeError(f"videos.zip is missing expected members: {missing[:5]}")
    members.sort(key=lambda item: item.video_name)
    return tuple(members)


def _extract_member(
    archive: zipfile.ZipFile,
    member: VideoArchiveMember,
    destination: Path,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite extracted video: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.part-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, archive.open(member.member_name) as source:
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != member.file_size:
            raise RuntimeError(f"Extracted video size mismatch: {member.member_name}")
        digest = sha256_file(temporary)
        temporary.replace(destination)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def probe_video(path: Path | str, *, ffprobe_binary: str = "ffprobe") -> VideoProbe:
    """Decode-count one video stream and return exact dimensions/frame count."""

    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,nb_frames,nb_read_frames",
        "-of",
        "json",
        str(Path(path)),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"ffprobe failed for {path}: {error}") from error
    try:
        payload = json.loads(completed.stdout)
        streams = payload["streams"]
        if len(streams) != 1:
            raise ValueError("expected one selected video stream")
        stream = streams[0]
        width = int(stream["width"])
        height = int(stream["height"])
        frame_count = next(
            int(value)
            for value in (stream.get("nb_read_frames"), stream.get("nb_frames"))
            if value not in {None, "", "N/A"}
        )
        codec = str(stream.get("codec_name") or "unknown")
    except (KeyError, StopIteration, TypeError, ValueError) as error:
        raise RuntimeError(f"Could not parse ffprobe output for {path}") from error
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video geometry or frame count for {path}")
    return VideoProbe(width=width, height=height, frame_count=frame_count, codec_name=codec)


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _select_ffmpeg_frame_sync_policy(help_text: str) -> FFmpegFrameSyncPolicy:
    """Select a passthrough spelling only from options ffmpeg advertises."""

    has_fps_mode = (
        re.search(r"(?m)^\s*-fps_mode(?:\[[^\]\r\n]*\])?(?:\s+|$)", help_text) is not None
    )
    has_vsync = re.search(r"(?m)^\s*-vsync(?:\s+|$)", help_text) is not None
    if has_fps_mode:
        return FFmpegFrameSyncPolicy(cli_option="-fps_mode", cli_value="passthrough")
    if has_vsync:
        # FFmpeg 4.4 exposes the older global spelling.  Numeric value 0 is
        # the documented alias for passthrough and is equivalent here because
        # this command has one selected video stream and one output.
        return FFmpegFrameSyncPolicy(cli_option="-vsync", cli_value="0")
    raise RuntimeError(
        "ffmpeg advertises neither -fps_mode nor -vsync; "
        "cannot guarantee one output frame per selected input frame"
    )


def resolve_ffmpeg_frame_sync_policy(
    ffmpeg_binary: str = "ffmpeg",
) -> FFmpegFrameSyncPolicy:
    """Resolve the supported passthrough spelling without decode-time fallback."""

    command = [ffmpeg_binary, "-hide_banner", "-h", "full"]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"Could not query {ffmpeg_binary} frame-sync capabilities: {error}"
        ) from error
    return _select_ffmpeg_frame_sync_policy(f"{completed.stdout}\n{completed.stderr}")


def iter_sampled_video_frames(
    path: Path | str,
    probe: VideoProbe,
    *,
    stride: int = VISUAL_WETLANDBIRDS_FRAME_STRIDE,
    offset: int = VISUAL_WETLANDBIRDS_FRAME_OFFSET,
    ffmpeg_binary: str = "ffmpeg",
    frame_sync_policy: FFmpegFrameSyncPolicy | None = None,
) -> Iterator[tuple[int, Image.Image]]:
    """Yield RGB frames selected by original decoded frame index."""

    if stride <= 0 or not 0 <= offset < stride:
        raise ValueError("stride must be positive and offset must be within one stride")
    if frame_sync_policy is None:
        frame_sync_policy = resolve_ffmpeg_frame_sync_policy(ffmpeg_binary)
    selected = tuple(range(offset, probe.frame_count, stride))
    expression = f"select=not(mod(n-{offset}\\,{stride}))"
    command = [
        ffmpeg_binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(Path(path)),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        expression,
        *frame_sync_policy.command_arguments(),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as error:
        raise RuntimeError(f"Could not launch ffmpeg for {path}: {error}") from error
    assert process.stdout is not None and process.stderr is not None
    frame_size = probe.width * probe.height * 3
    try:
        for frame_index in selected:
            payload = _read_exact(process.stdout, frame_size)
            if len(payload) != frame_size:
                error_text = process.stderr.read().decode("utf-8", errors="replace")
                process.wait(timeout=30)
                raise RuntimeError(
                    f"ffmpeg returned a short frame for {path} at {frame_index}: {error_text}"
                )
            yield frame_index, Image.frombytes("RGB", (probe.width, probe.height), payload)
        extra = process.stdout.read(1)
        error_text = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait(timeout=120)
        if extra or return_code != 0:
            raise RuntimeError(
                f"ffmpeg frame extraction failed for {path} "
                f"(return={return_code}, extra_output={bool(extra)}): {error_text}"
            )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - pathological decoder.
                process.kill()
                process.wait(timeout=10)


def _binary_version(binary: str) -> str:
    try:
        completed = subprocess.run(
            [binary, "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Could not query {binary} version: {error}") from error
    first_line = completed.stdout.splitlines()
    if not first_line:
        raise RuntimeError(f"Empty {binary} version output")
    return first_line[0]


def _sample_uri(annotation: WetlandAnnotation, member: VideoArchiveMember) -> str:
    fragment = urlencode(
        {
            "member": member.member_name,
            "frame": annotation.frame_index,
            "bird_id": annotation.bird_id,
        }
    )
    return f"{VISUAL_WETLANDBIRDS_VIDEO_URL}#{fragment}"


def visual_wetlandbirds_group_id(video_name: str, bird_id: int) -> str:
    """Return the stable original-video/track group used for split auditing."""

    if not _VIDEO_NAME_RE.fullmatch(video_name):
        raise ValueError(f"Invalid Visual WetlandBirds video name: {video_name!r}")
    if not isinstance(bird_id, int) or isinstance(bird_id, bool) or bird_id < 0:
        raise ValueError("bird_id must be a non-negative integer")
    return f"visual-wetlandbirds-video-track:{video_name}:bird-{bird_id}"


def _crop_relative_path(annotation: WetlandAnnotation, split: str, taxon: BirdTaxon) -> str:
    birdnet_id = taxon.taxon_id.removeprefix("birdnet:")
    return (
        f"crops/{split}/{birdnet_id}/{annotation.video_name}/"
        f"f{annotation.frame_index:06d}-bird{annotation.bird_id:03d}.png"
    )


def _prepare_visual_wetlandbirds_locked(
    root: Path | str,
    *,
    birdnet_csv_path: Path | str,
    download_metadata: bool = True,
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str = "ffprobe",
    progress: ProgressCallback | None = None,
) -> VisualWetlandBirdsPreparation:
    """Build the immutable stride-10 crop source from an existing videos.zip.

    Missing small metadata files may be fetched from Zenodo.  The 9.4 GB video
    archive is never downloaded by this function and must already exist at
    ``ROOT/videos.zip``.
    """

    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    final_crops = root_path / "crops"
    final_manifests = root_path / "manifests"
    if final_crops.exists() or final_manifests.exists():
        raise FileExistsError("Refusing to replace existing Visual WetlandBirds crops or manifests")
    _notify(progress, phase="metadata", status="start")
    metadata_md5 = ensure_official_metadata(root_path, download_missing=download_metadata)
    species_by_id = read_species_ids(root_path / "species_ID.csv")
    behavior_by_id = read_behavior_ids(root_path / "behaviors_ID.csv")
    annotations, annotated_frame_rows = read_bounding_box_annotations(
        root_path / "bounding_boxes.csv",
        species_by_id=species_by_id,
        behavior_by_id=behavior_by_id,
    )
    annotation_audit = audit_annotation_counts(annotations)
    video_names = tuple(sorted({annotation.video_name for annotation in annotations}))
    video_species_id: dict[str, int] = {}
    for annotation in annotations:
        video_species_id.setdefault(annotation.video_name, annotation.species_id)
    splits = read_official_splits(root_path / "splits.json", expected_video_names=video_names)
    behavior_clip_count = read_behavior_clip_count(
        root_path / "crops.csv",
        video_species_id=video_species_id,
        behavior_by_id=behavior_by_id,
    )
    if (
        len(species_by_id) != _EXPECTED_TAXON_COUNT
        or len(behavior_by_id) != 7
        or len(video_names) != _EXPECTED_VIDEO_COUNT
        or annotated_frame_rows != _EXPECTED_ANNOTATED_FRAME_ROWS
        or len(annotations) != _EXPECTED_DENSE_BOX_COUNT
        or behavior_clip_count != _EXPECTED_BEHAVIOR_CLIP_COUNT
        or annotation_audit.sampled_bbox_count != _EXPECTED_SAMPLED_CROP_COUNT
    ):
        raise RuntimeError(
            "Visual WetlandBirds v4 official counts changed: "
            f"taxa={len(species_by_id)}, behaviors={len(behavior_by_id)}, "
            f"videos={len(video_names)}, frame_rows={annotated_frame_rows}, "
            f"boxes={len(annotations)}, clips={behavior_clip_count}"
        )
    taxa, taxon_by_source_id = build_visual_wetlandbirds_taxa(species_by_id, birdnet_csv_path)
    birdnet_csv_sha256 = sha256_file(Path(birdnet_csv_path).expanduser())
    _notify(progress, phase="metadata", status="complete")

    _notify(progress, phase="archive_checksum_and_listing", status="start")
    archive_path = root_path / "videos.zip"
    archive_members = validate_video_archive(
        archive_path,
        expected_video_names=video_names,
    )
    _notify(progress, phase="archive_checksum_and_listing", status="complete")

    by_video_frame: dict[str, dict[int, list[WetlandAnnotation]]] = defaultdict(
        lambda: defaultdict(list)
    )
    dense_by_video: Counter[str] = Counter()
    frame_rows_by_video: Counter[str] = Counter()
    max_frame_by_video: dict[str, int] = {}
    for annotation in annotations:
        dense_by_video[annotation.video_name] += 1
        max_frame_by_video[annotation.video_name] = max(
            max_frame_by_video.get(annotation.video_name, -1), annotation.frame_index
        )
        if (
            annotation.frame_index % VISUAL_WETLANDBIRDS_FRAME_STRIDE
            == VISUAL_WETLANDBIRDS_FRAME_OFFSET
        ):
            by_video_frame[annotation.video_name][annotation.frame_index].append(annotation)
    for video, _frame in {(item.video_name, item.frame_index) for item in annotations}:
        frame_rows_by_video[video] += 1
    sampled_annotations = tuple(
        annotation
        for annotation in annotations
        if annotation.frame_index % VISUAL_WETLANDBIRDS_FRAME_STRIDE
        == VISUAL_WETLANDBIRDS_FRAME_OFFSET
    )
    sampled_frame_rows = annotation_audit.sampled_annotated_frame_rows

    ffmpeg_version = _binary_version(ffmpeg_binary)
    frame_sync_policy = resolve_ffmpeg_frame_sync_policy(ffmpeg_binary)
    ffprobe_version = _binary_version(ffprobe_binary)
    stage = Path(tempfile.mkdtemp(prefix=".visual-wetlandbirds-prepare-", dir=root_path))
    stage_crops = stage / "crops"
    stage_manifests = stage / "manifests"
    scratch = stage / "scratch"
    samples: list[BirdSample] = []
    provenance: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    decoded_frame_count = 0
    decoded_sampled_frame_count = 0
    try:
        member_by_video = {member.video_name: member for member in archive_members}
        with zipfile.ZipFile(archive_path) as archive:
            for index, video_name in enumerate(video_names, start=1):
                _notify(
                    progress,
                    phase="video",
                    index=index,
                    total=len(video_names),
                    video_name=video_name,
                    status="start",
                )
                member = member_by_video[video_name]
                video_path = scratch / f"{video_name}.mp4"
                extracted_sha256 = _extract_member(archive, member, video_path)
                probe = probe_video(video_path, ffprobe_binary=ffprobe_binary)
                max_annotated_frame = max_frame_by_video[video_name]
                if probe.frame_count != max_annotated_frame + 1:
                    raise RuntimeError(
                        f"Video/annotation frame-count mismatch for {video_name}: "
                        f"video={probe.frame_count}, max_annotation={max_annotated_frame}"
                    )
                decoded_frame_count += probe.frame_count
                sampled_decoded_for_video = 0
                sampled_crops_for_video = 0
                clipped_boxes_for_video = 0
                for frame_index, frame in iter_sampled_video_frames(
                    video_path,
                    probe,
                    ffmpeg_binary=ffmpeg_binary,
                    frame_sync_policy=frame_sync_policy,
                ):
                    sampled_decoded_for_video += 1
                    for annotation in by_video_frame[video_name].get(frame_index, ()):
                        effective_bbox, bbox_was_clipped = clip_bbox_xyxy_to_image(
                            annotation.bbox_xyxy, frame.size
                        )
                        clipped_boxes_for_video += int(bbox_was_clipped)
                        crop, geometry = square_context_crop(
                            frame,
                            effective_bbox,
                            context_scale=VISUAL_WETLANDBIRDS_CONTEXT_SCALE,
                        )
                        taxon = taxon_by_source_id[annotation.species_id]
                        split = splits[video_name]
                        relative_path = _crop_relative_path(annotation, split, taxon)
                        digest, phash = save_crop_png_atomic(crop, stage / relative_path)
                        source_sample_id = (
                            f"{video_name}:frame-{annotation.frame_index}:bird-{annotation.bird_id}"
                        )
                        samples.append(
                            BirdSample(
                                dataset_id=VISUAL_WETLANDBIRDS_DATASET_ID,
                                source_sample_id=source_sample_id,
                                source_split=split,
                                relative_path=relative_path,
                                image_uri=_sample_uri(annotation, member),
                                group_id=visual_wetlandbirds_group_id(
                                    video_name, annotation.bird_id
                                ),
                                raw_label=annotation.species,
                                taxon_id=taxon.taxon_id,
                                sha256=digest,
                                phash=phash,
                                license=VISUAL_WETLANDBIRDS_LICENSE,
                                author=VISUAL_WETLANDBIRDS_AUTHORS,
                                source=(
                                    f"Visual WetlandBirds v4, Zenodo {VISUAL_WETLANDBIRDS_DOI}"
                                ),
                            )
                        )
                        provenance.append(
                            {
                                "source_sample_id": source_sample_id,
                                "archive_uri": VISUAL_WETLANDBIRDS_VIDEO_URL,
                                "archive_md5": VISUAL_WETLANDBIRDS_VIDEO_MD5,
                                "archive_member": member.member_name,
                                "archive_member_crc32": member.crc32,
                                "archive_member_size": member.file_size,
                                "extracted_video_sha256": extracted_sha256,
                                "annotation_uri": _metadata_url("bounding_boxes.csv"),
                                "annotation_md5": metadata_md5["bounding_boxes.csv"],
                                "frame_index": annotation.frame_index,
                                "bird_id": annotation.bird_id,
                                "behavior_id": annotation.behavior_id,
                                "source_bbox_xyxy": annotation.bbox_xyxy,
                                "effective_bbox_xyxy": effective_bbox,
                                "bbox_was_clipped": bbox_was_clipped,
                                "crop_geometry": geometry.to_dict(),
                                "crop_sha256": digest,
                                "crop_phash": phash,
                            }
                        )
                        sampled_crops_for_video += 1
                decoded_sampled_frame_count += sampled_decoded_for_video
                video_rows.append(
                    {
                        "video_name": video_name,
                        "source_split": splits[video_name],
                        "source_species_id": video_species_id[video_name],
                        "source_species": species_by_id[video_species_id[video_name]],
                        "archive_uri": VISUAL_WETLANDBIRDS_VIDEO_URL,
                        "archive_md5": VISUAL_WETLANDBIRDS_VIDEO_MD5,
                        "archive_member": member.member_name,
                        "archive_member_crc32": member.crc32,
                        "archive_member_size": member.file_size,
                        "archive_member_compressed_size": member.compressed_size,
                        "extracted_video_sha256": extracted_sha256,
                        "codec_name": probe.codec_name,
                        "width": probe.width,
                        "height": probe.height,
                        "frame_count": probe.frame_count,
                        "max_annotated_frame": max_annotated_frame,
                        "annotated_frame_rows": frame_rows_by_video[video_name],
                        "dense_bbox_count": dense_by_video[video_name],
                        "sampled_decoded_frame_count": sampled_decoded_for_video,
                        "sampled_crop_count": sampled_crops_for_video,
                        "clipped_bbox_count": clipped_boxes_for_video,
                    }
                )
                video_path.unlink()
                _notify(
                    progress,
                    phase="video",
                    index=index,
                    total=len(video_names),
                    video_name=video_name,
                    status="complete",
                    sampled_crops=sampled_crops_for_video,
                )

        samples.sort(key=lambda item: item.source_sample_id)
        provenance.sort(key=lambda item: item["source_sample_id"])
        if len(samples) != len(sampled_annotations) or len(provenance) != len(samples):
            raise RuntimeError("Decoded crop count does not match stride-10 annotations")
        split_video_counts = Counter(splits.values())
        split_annotated_frame_counts = Counter(
            splits[video]
            for video, _frame in {(item.video_name, item.frame_index) for item in annotations}
        )
        split_dense_bbox_counts = Counter(splits[item.video_name] for item in annotations)
        split_crop_counts = Counter(sample.source_split for sample in samples)
        behavior_id_counts = Counter(item.behavior_id for item in annotations)
        report = VisualWetlandBirdsPreparation(
            video_archive_md5=VISUAL_WETLANDBIRDS_VIDEO_MD5,
            metadata_md5=metadata_md5,
            birdnet_csv_sha256=birdnet_csv_sha256,
            video_count=len(video_names),
            taxon_count=len(taxa),
            behavior_clip_count=behavior_clip_count,
            annotated_frame_rows=annotated_frame_rows,
            dense_bbox_count=len(annotations),
            sampled_annotated_frame_rows=sampled_frame_rows,
            sampled_crop_count=len(samples),
            decoded_frame_count=decoded_frame_count,
            decoded_sampled_frame_count=decoded_sampled_frame_count,
            split_video_counts=dict(sorted(split_video_counts.items())),
            split_annotated_frame_counts=dict(sorted(split_annotated_frame_counts.items())),
            split_dense_bbox_counts=dict(sorted(split_dense_bbox_counts.items())),
            split_crop_counts=dict(sorted(split_crop_counts.items())),
            behavior_id_counts={
                str(key): value for key, value in sorted(behavior_id_counts.items())
            },
            unmapped_behavior_box_count=sum(
                behavior_id_counts[identifier]
                for identifier in VISUAL_WETLANDBIRDS_UNMAPPED_BEHAVIOR_IDS
            ),
        )
        write_jsonl(stage_manifests / "taxa.jsonl", (asdict(taxon) for taxon in taxa))
        write_jsonl(stage_manifests / "samples.jsonl", (asdict(sample) for sample in samples))
        write_jsonl(stage_manifests / "sample_provenance.jsonl", provenance)
        write_jsonl(stage_manifests / "videos.jsonl", video_rows)
        source_manifest = {
            "dataset_id": VISUAL_WETLANDBIRDS_DATASET_ID,
            "dataset_role": "auxiliary_training_source_from_correlated_video_crops",
            "source_url": VISUAL_WETLANDBIRDS_SOURCE_URL,
            "doi": VISUAL_WETLANDBIRDS_DOI,
            "record_version": VISUAL_WETLANDBIRDS_VERSION,
            "video_archive_url": VISUAL_WETLANDBIRDS_VIDEO_URL,
            "license": VISUAL_WETLANDBIRDS_LICENSE,
            "creators": VISUAL_WETLANDBIRDS_AUTHORS,
            "crop_protocol": {
                "protocol": VISUAL_WETLANDBIRDS_CROP_PROTOCOL,
                "paper_frame_downsample": VISUAL_WETLANDBIRDS_FRAME_STRIDE,
                "frame_offset": VISUAL_WETLANDBIRDS_FRAME_OFFSET,
                "stride_domain": annotation_audit.stride_domain,
                "frame_selection": "first decoded frame in each consecutive group of ten",
                "dense_bbox_count": annotation_audit.dense_bbox_count,
                "kept_bbox_count": annotation_audit.sampled_bbox_count,
                "square_crop_recipe": SQUARE_CONTEXT_RECIPE_ID,
                "context_scale": VISUAL_WETLANDBIRDS_CONTEXT_SCALE,
                "target_exclusions_applied": False,
            },
            "taxonomy_alignment": {
                "taxonomy_source": BIRDNET_TAXONOMY_AUTHORITY,
                "taxonomy_version": BIRDNET_TAXONOMY_VERSION,
                "birdnet_csv_sha256": birdnet_csv_sha256,
                "alignments": [asdict(row) for row in VISUAL_WETLANDBIRDS_TAXON_ALIGNMENTS],
            },
            "decoder": {
                "ffmpeg": ffmpeg_version,
                "ffprobe": ffprobe_version,
                "frame_sync": frame_sync_policy.to_dict(),
                "pixel_format": "rgb24",
                "pillow": PILLOW_VERSION,
            },
            "publisher_annotation_audit": {
                "defined_behavior_ids": sorted(behavior_by_id),
                "unmapped_behavior_ids_preserved": list(VISUAL_WETLANDBIRDS_UNMAPPED_BEHAVIOR_IDS),
                "behavior_id_counts": report.behavior_id_counts,
                "note": (
                    "v4 bounding_boxes.csv uses id 7 although behaviors_ID.csv "
                    "defines only 0--6; species labels/crops do not depend on behavior."
                ),
            },
            "report": report.to_dict(),
        }
        write_json(stage_manifests / "source.json", source_manifest)

        stage_crops.rename(final_crops)
        try:
            stage_manifests.rename(final_manifests)
        except Exception:
            final_crops.rename(stage_crops)
            raise
        return report
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def prepare_visual_wetlandbirds(
    root: Path | str,
    *,
    birdnet_csv_path: Path | str,
    download_metadata: bool = True,
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str = "ffprobe",
    progress: ProgressCallback | None = None,
) -> VisualWetlandBirdsPreparation:
    """Prepare one source under a non-blocking, process-safe dataset lock."""

    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    lock_path = root_path / ".visual-wetlandbirds.prepare.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another Visual WetlandBirds preparation holds {lock_path}"
            ) from error
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        try:
            return _prepare_visual_wetlandbirds_locked(
                root_path,
                birdnet_csv_path=birdnet_csv_path,
                download_metadata=download_metadata,
                ffmpeg_binary=ffmpeg_binary,
                ffprobe_binary=ffprobe_binary,
                progress=progress,
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_visual_wetlandbirds_manifests(
    root: Path | str,
) -> tuple[tuple[BirdSample, ...], tuple[BirdTaxon, ...]]:
    """Load completed canonical manifests without reading videos.zip."""

    from .bird_manifest import load_samples, load_taxa

    root_path = Path(root).expanduser()
    return (
        load_samples(root_path / "manifests/samples.jsonl"),
        load_taxa(root_path / "manifests/taxa.jsonl"),
    )
