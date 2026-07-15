"""Audited preparation of the official USGS aerial-avian crop release.

The publisher already provides individual bird crops, so this preparer never
invents a bounding box or re-crops a source image.  It verifies the paired
``annotations.zip`` and ``images.zip`` releases, keeps the official
train/validation/test membership, removes only the two explicitly coarse
labels (``scoter spp`` and ``non-target Species``), and records that upstream
crop provenance in an immutable manifest bundle.

CUB target taxa are retained in the canonical source data.  A strict-CUB audit
is available separately so experiment-specific exclusions cannot silently
change the source manifest.
"""

from __future__ import annotations

import ast
import re
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from .bird_crops import md5_file, validate_relative_posix_path
from .bird_manifest import (
    BirdSample,
    BirdTaxon,
    perceptual_dhash,
    sha256_file,
    validate_manifest,
)
from .bird_source_archive import (
    contained_output_path,
    extract_zip_member_no_replace,
    publish_manifest_bundle,
    stream_download_verified,
    validated_zip_members,
)
from .birdnet_lock import BirdNetTaxonLock, resolve_locked_birdnet_taxa

USGS_AERIAL_AVIAN_DATASET_ID = "usgs-aerial-avian-2023-publisher-crops"
USGS_AERIAL_AVIAN_VERSION = "2023-06-29"
USGS_AERIAL_AVIAN_SOURCE_URL = (
    "https://www.usgs.gov/data/images-and-annotations-automate-classification-avian-species"
)
USGS_AERIAL_AVIAN_DOI = "https://doi.org/10.5066/P9YL80R6"
USGS_AERIAL_AVIAN_LICENSE = "CC0 1.0 Universal"
USGS_AERIAL_AVIAN_LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
USGS_AERIAL_AVIAN_AUTHOR = (
    "Miao, Fara, Fronczak, Landolt, Bragger, Koneff, Lubinski, Robinson, and Yates"
)
USGS_ANNOTATIONS_URL = (
    "https://www.sciencebase.gov/catalog/file/get/63c87180d34e06fef14f353f"
    "?f=__disk__66%2F45%2F7b%2F66457b962e7867e174ff1ec37e6d4cfd1d7ece25"
)
USGS_IMAGES_URL = (
    "https://www.sciencebase.gov/catalog/file/get/63c87661d34e06fef14f355d"
    "?f=__disk__63%2F71%2F0e%2F63710e13d978692cb852e39ac88aa7a582708579"
)
USGS_ANNOTATIONS_MD5 = "c3c702e9b19c127bc4edde1ee21f8d45"
USGS_IMAGES_MD5 = "7b059330471e1d2bf032d1d8871d47cb"
USGS_ANNOTATIONS_SIZE = 45_939
USGS_IMAGES_SIZE = 178_402_490
USGS_BIRDNET_CSV_SHA256 = "37b4015719c0c9e014d3c994dd188904457ff74edc3e3002a01b57fa9830d426"

_SCIENCEBASE_HOSTS = {"www.sciencebase.gov"}
_ANNOTATION_PREFIX = "annotations/"
_ANNOTATION_FILES = {
    "annotations/labelmap.txt",
    "annotations/train.txt",
    "annotations/val.txt",
    "annotations/test.txt",
}
_SPLIT_MEMBERS = {
    "annotations/train.txt": "train",
    "annotations/val.txt": "validation",
    "annotations/test.txt": "test",
}
_EXPECTED_LABELMAP = {
    1: "Common Eider",
    2: "Long-tailed Duck",
    3: "scoter spp",
    4: "Black Scoter",
    5: "non-target Species",
    6: "White-winged Scoter",
    7: "Gull",
    8: "Surf Scoter",
    9: "King Eider",
    10: "Bufflehead",
}
_SPECIFIC_CATEGORY_IDS = frozenset({1, 2, 4, 6})
_COARSE_CATEGORY_IDS = frozenset({3, 5})
_EXPECTED_COUNTS = {
    "all_rows": 10_917,
    "specific_rows": 10_186,
    "coarse_rows": 731,
    "taxa": 4,
    "source_groups": 67,
}
_EXPECTED_ALL_SPLIT_COUNTS = {"train": 7_223, "validation": 3_458, "test": 236}
_EXPECTED_SPECIFIC_SPLIT_COUNTS = {
    "train": 6_649,
    "validation": 3_306,
    "test": 231,
}
_EXPECTED_USED_CATEGORY_COUNTS = {1: 9_418, 2: 253, 3: 580, 4: 449, 5: 151, 6: 66}
_EXPECTED_PATH_LABEL_MISMATCHES = {(3, 8): 4, (5, 10): 1}
_FILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.png$")
_OBJECT_SUFFIX_RE = re.compile(r"^(?P<frame>.+)_(?P<object>[0-9]+)$")

USGS_AERIAL_AVIAN_TAXON_LOCKS: tuple[BirdNetTaxonLock, ...] = (
    BirdNetTaxonLock("Common Eider", "BN14015", "Somateria mollissima", "Common Eider"),
    BirdNetTaxonLock("Long-tailed Duck", "BN03328", "Clangula hyemalis", "Long-tailed Duck"),
    BirdNetTaxonLock("Black Scoter", "BN08586", "Melanitta americana", "Black Scoter"),
    BirdNetTaxonLock(
        "White-winged Scoter",
        "BN08587",
        "Melanitta deglandi",
        "White-winged Scoter",
    ),
)


@dataclass(frozen=True, slots=True)
class UsgsAerialRow:
    source_sample_id: str
    source_split: str
    relative_path: str
    encoded_label: int
    category_id: int
    publisher_path_category_id: int
    raw_label: str
    taxon_id: str | None
    source_frame_id: str


@dataclass(frozen=True, slots=True)
class UsgsAerialPlan:
    annotations_path: Path
    annotations_md5: str
    birdnet_csv_sha256: str
    all_rows: tuple[UsgsAerialRow, ...]
    specific_rows: tuple[UsgsAerialRow, ...]
    taxa: tuple[BirdTaxon, ...]
    all_split_counts: dict[str, int]
    specific_split_counts: dict[str, int]
    category_counts: dict[int, int]


@dataclass(frozen=True, slots=True)
class UsgsAerialPreparation:
    annotations_md5: str
    images_md5: str
    birdnet_csv_sha256: str
    publisher_crop_count: int
    extracted_crops: int
    reused_crops: int
    omitted_coarse_crops: int
    taxon_count: int
    source_group_count: int
    split_counts: dict[str, int]
    manifest_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UsgsAerialArchiveAudit:
    annotations_md5: str
    images_md5: str
    annotation_rows: int
    image_members: int
    specific_members: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UsgsAerialStrictCubAudit:
    total_crops: int
    total_taxa: int
    excluded_crops: int
    excluded_taxa: int
    retained_crops: int
    retained_taxa: int
    excluded_counts_by_taxon: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["excluded_counts_by_taxon"] = dict(self.excluded_counts_by_taxon)
        return value


def _decode_utf8(bundle: zipfile.ZipFile, member: str) -> str:
    try:
        return bundle.read(member).decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"USGS annotation member is not UTF-8: {member}") from error


def _parse_labelmap(value: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for line_number, line in enumerate(value.splitlines(), start=1):
        if not line.strip() or line != line.strip():
            raise RuntimeError(f"Malformed USGS labelmap line {line_number}")
        try:
            row = ast.literal_eval(line)
        except (SyntaxError, ValueError) as error:
            raise RuntimeError(f"Malformed USGS labelmap line {line_number}") from error
        if not isinstance(row, dict) or set(row) != {"id", "name", "supercategory"}:
            raise RuntimeError(f"Unexpected USGS labelmap schema at line {line_number}")
        category_id = row["id"]
        name = row["name"]
        if (
            isinstance(category_id, bool)
            or not isinstance(category_id, int)
            or category_id <= 0
            or not isinstance(name, str)
            or not name.strip()
            or row["supercategory"] != ""
        ):
            raise RuntimeError(f"Invalid USGS labelmap row at line {line_number}")
        if category_id in result:
            raise RuntimeError(f"Duplicate USGS category id: {category_id}")
        result[category_id] = name
    if not result:
        raise RuntimeError("USGS labelmap must not be empty")
    return result


def _parse_annotation_line(
    line: str,
    *,
    line_number: int,
    source_split: str,
    labelmap: dict[int, str],
) -> UsgsAerialRow:
    if not line or line != line.strip() or line.count(" ") != 1:
        raise RuntimeError(f"Malformed USGS annotation line {line_number} ({source_split})")
    path_value, encoded_value = line.rsplit(" ", 1)
    validate_relative_posix_path(path_value)
    try:
        encoded_label = int(encoded_value)
    except ValueError as error:
        raise RuntimeError(
            f"Non-integer USGS label at line {line_number} ({source_split})"
        ) from error
    if str(encoded_label) != encoded_value or encoded_label < 0:
        raise RuntimeError(f"Non-canonical USGS label at line {line_number} ({source_split})")
    category_id = encoded_label + 1
    raw_label = labelmap.get(category_id)
    if raw_label is None:
        raise RuntimeError(f"USGS annotation references unknown label: {encoded_label}")
    parts = PurePosixPath(path_value).parts
    if (
        len(parts) != 4
        or parts[0] != "images"
        or parts[1] not in {"train", "eval", "test"}
        or not _FILE_NAME_RE.fullmatch(parts[3])
    ):
        raise RuntimeError(f"Unexpected USGS image path schema: {path_value}")
    try:
        publisher_path_category_id = int(parts[2])
    except ValueError as error:
        raise RuntimeError(f"Non-integer USGS path category: {path_value}") from error
    if str(publisher_path_category_id) != parts[2] or publisher_path_category_id not in labelmap:
        raise RuntimeError(f"Unknown USGS path category: {path_value}")
    match = _OBJECT_SUFFIX_RE.fullmatch(PurePosixPath(path_value).stem)
    if match is None or not match.group("frame"):
        raise RuntimeError(f"USGS crop name lacks a source-frame/object suffix: {path_value}")
    return UsgsAerialRow(
        source_sample_id=path_value,
        source_split=source_split,
        relative_path=path_value,
        encoded_label=encoded_label,
        category_id=category_id,
        publisher_path_category_id=publisher_path_category_id,
        raw_label=raw_label,
        taxon_id=None,
        source_frame_id=match.group("frame"),
    )


def plan_usgs_aerial_avian(
    annotations_path: Path | str,
    *,
    birdnet_csv_path: Path | str,
    require_official_lock: bool = True,
) -> UsgsAerialPlan:
    """Validate official annotation schema/taxonomy before the image archive is used."""

    annotations = Path(annotations_path).expanduser()
    if annotations.is_symlink() or not annotations.is_file():
        raise FileNotFoundError(f"Missing regular USGS annotations archive: {annotations}")
    annotations_md5 = md5_file(annotations)
    if require_official_lock and (
        annotations_md5 != USGS_ANNOTATIONS_MD5
        or annotations.stat().st_size != USGS_ANNOTATIONS_SIZE
    ):
        raise RuntimeError(
            "USGS annotations archive differs from the official release: "
            f"size={annotations.stat().st_size}, md5={annotations_md5}"
        )
    validated_zip_members(annotations, expected_files=_ANNOTATION_FILES)
    with zipfile.ZipFile(annotations) as bundle:
        labelmap = _parse_labelmap(_decode_utf8(bundle, "annotations/labelmap.txt"))
        rows: list[UsgsAerialRow] = []
        for member, source_split in _SPLIT_MEMBERS.items():
            text = _decode_utf8(bundle, member)
            lines = text.splitlines()
            if not lines:
                raise RuntimeError(f"USGS split annotation is empty: {member}")
            rows.extend(
                _parse_annotation_line(
                    line,
                    line_number=line_number,
                    source_split=source_split,
                    labelmap=labelmap,
                )
                for line_number, line in enumerate(lines, start=1)
            )
    sample_ids = [row.source_sample_id for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("USGS annotations contain duplicate crop paths")
    category_counts = Counter(row.category_id for row in rows)
    used_categories = set(category_counts)
    expected_used = _SPECIFIC_CATEGORY_IDS | _COARSE_CATEGORY_IDS
    if used_categories != expected_used:
        raise RuntimeError(
            f"Unexpected used USGS categories: {sorted(used_categories)}; "
            f"expected {sorted(expected_used)}"
        )
    if require_official_lock and labelmap != _EXPECTED_LABELMAP:
        raise RuntimeError(f"USGS labelmap changed: {labelmap}")

    taxa_by_label, birdnet_digest = resolve_locked_birdnet_taxa(
        birdnet_csv_path,
        USGS_AERIAL_AVIAN_TAXON_LOCKS,
        expected_sha256=USGS_BIRDNET_CSV_SHA256,
        require_official_lock=require_official_lock,
    )
    all_rows = tuple(sorted(rows, key=lambda row: row.source_sample_id))
    specific_rows = tuple(
        UsgsAerialRow(
            source_sample_id=row.source_sample_id,
            source_split=row.source_split,
            relative_path=row.relative_path,
            encoded_label=row.encoded_label,
            category_id=row.category_id,
            publisher_path_category_id=row.publisher_path_category_id,
            raw_label=row.raw_label,
            taxon_id=taxa_by_label[row.raw_label].taxon_id,
            source_frame_id=row.source_frame_id,
        )
        for row in all_rows
        if row.category_id in _SPECIFIC_CATEGORY_IDS
    )
    all_split_counts = dict(sorted(Counter(row.source_split for row in all_rows).items()))
    specific_split_counts = dict(sorted(Counter(row.source_split for row in specific_rows).items()))
    taxa = tuple(sorted(taxa_by_label.values(), key=lambda taxon: taxon.taxon_id))
    observed_counts = {
        "all_rows": len(all_rows),
        "specific_rows": len(specific_rows),
        "coarse_rows": len(all_rows) - len(specific_rows),
        "taxa": len(taxa),
        "source_groups": len({row.source_frame_id for row in all_rows}),
    }
    path_label_mismatches = dict(
        sorted(
            Counter(
                (row.category_id, row.publisher_path_category_id)
                for row in all_rows
                if row.category_id != row.publisher_path_category_id
            ).items()
        )
    )
    if require_official_lock and (
        observed_counts != _EXPECTED_COUNTS
        or all_split_counts != _EXPECTED_ALL_SPLIT_COUNTS
        or specific_split_counts != _EXPECTED_SPECIFIC_SPLIT_COUNTS
        or dict(sorted(category_counts.items())) != _EXPECTED_USED_CATEGORY_COUNTS
        or path_label_mismatches != _EXPECTED_PATH_LABEL_MISMATCHES
    ):
        raise RuntimeError(
            "Unexpected USGS official counts: "
            f"counts={observed_counts}, all_splits={all_split_counts}, "
            f"specific_splits={specific_split_counts}, categories={dict(category_counts)}"
            f", path_label_mismatches={path_label_mismatches}"
        )
    return UsgsAerialPlan(
        annotations_path=annotations.resolve(),
        annotations_md5=annotations_md5,
        birdnet_csv_sha256=birdnet_digest,
        all_rows=all_rows,
        specific_rows=specific_rows,
        taxa=taxa,
        all_split_counts=all_split_counts,
        specific_split_counts=specific_split_counts,
        category_counts=dict(sorted(category_counts.items())),
    )


def download_usgs_aerial_annotations(
    root: Path | str = Path("data/usgs_aerial_avian"),
) -> Path:
    """Download and checksum-lock only the 45-KB official annotations archive."""

    root_path = Path(root).expanduser().resolve()
    destination = contained_output_path(root_path, "sources/annotations.zip")
    stream_download_verified(
        USGS_ANNOTATIONS_URL,
        destination,
        expected_md5=USGS_ANNOTATIONS_MD5,
        expected_size=USGS_ANNOTATIONS_SIZE,
        allowed_hosts=_SCIENCEBASE_HOSTS,
    )
    return destination


def _ensure_images_archive(root: Path) -> Path:
    destination = contained_output_path(root, "sources/images.zip")
    stream_download_verified(
        USGS_IMAGES_URL,
        destination,
        expected_md5=USGS_IMAGES_MD5,
        expected_size=USGS_IMAGES_SIZE,
        allowed_hosts=_SCIENCEBASE_HOSTS,
    )
    return destination


def audit_usgs_aerial_archives(
    plan: UsgsAerialPlan,
    images_path: Path | str,
    *,
    require_official_lock: bool = True,
) -> UsgsAerialArchiveAudit:
    """Verify the image archive checksum and exact annotation/image pairing."""

    images = Path(images_path).expanduser()
    if images.is_symlink() or not images.is_file():
        raise FileNotFoundError(f"Missing regular USGS images archive: {images}")
    images_md5 = md5_file(images)
    if require_official_lock and (
        images_md5 != USGS_IMAGES_MD5 or images.stat().st_size != USGS_IMAGES_SIZE
    ):
        raise RuntimeError(
            "USGS images archive differs from the official release: "
            f"size={images.stat().st_size}, md5={images_md5}"
        )
    members = validated_zip_members(
        images,
        expected_files=(row.relative_path for row in plan.all_rows),
    )
    return UsgsAerialArchiveAudit(
        annotations_md5=plan.annotations_md5,
        images_md5=images_md5,
        annotation_rows=len(plan.all_rows),
        image_members=sum(not info.is_dir() for info in members.values()),
        specific_members=len(plan.specific_rows),
    )


def audit_usgs_aerial_strict_cub(
    rows: Sequence[UsgsAerialRow | BirdSample],
    excluded_birdnet_ids: Iterable[str],
) -> UsgsAerialStrictCubAudit:
    """Report target exclusions without mutating the canonical source rows."""

    if not rows:
        raise ValueError("USGS strict-CUB audit rows must not be empty")
    excluded = {
        value if value.startswith("birdnet:") else f"birdnet:{value}"
        for raw in excluded_birdnet_ids
        if (value := str(raw).strip())
    }
    counts = Counter(row.taxon_id for row in rows)
    if None in counts:
        raise ValueError("USGS strict-CUB audit requires taxonomy-resolved rows")
    excluded_counts = {key: value for key, value in counts.items() if key in excluded}
    return UsgsAerialStrictCubAudit(
        total_crops=len(rows),
        total_taxa=len(counts),
        excluded_crops=sum(excluded_counts.values()),
        excluded_taxa=len(excluded_counts),
        retained_crops=len(rows) - sum(excluded_counts.values()),
        retained_taxa=len(counts) - len(excluded_counts),
        excluded_counts_by_taxon=tuple(sorted(excluded_counts.items())),
    )


def prepare_usgs_aerial_avian(
    root: Path | str = Path("data/usgs_aerial_avian"),
    *,
    birdnet_csv_path: Path | str,
    annotations_path: Path | str | None = None,
    images_path: Path | str | None = None,
    require_official_lock: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> UsgsAerialPreparation:
    """Verify both releases and materialize the 10,186 species-level publisher crops."""

    root_path = Path(root).expanduser().resolve()
    manifests = root_path / "manifests"
    if manifests.exists() or manifests.is_symlink():
        raise FileExistsError(f"Refusing to replace manifest directory: {manifests}")
    if not require_official_lock and (annotations_path is None or images_path is None):
        raise ValueError("non-official preparation requires explicit archive paths")
    annotations = (
        download_usgs_aerial_annotations(root_path)
        if annotations_path is None
        else Path(annotations_path).expanduser()
    )
    images = (
        _ensure_images_archive(root_path) if images_path is None else Path(images_path).expanduser()
    )
    plan = plan_usgs_aerial_avian(
        annotations,
        birdnet_csv_path=birdnet_csv_path,
        require_official_lock=require_official_lock,
    )
    archive_audit = audit_usgs_aerial_archives(
        plan,
        images,
        require_official_lock=require_official_lock,
    )
    members = validated_zip_members(images)
    samples: list[BirdSample] = []
    provenance: list[dict[str, Any]] = []
    extracted = 0
    reused = 0
    total = len(plan.specific_rows)
    for completed, row in enumerate(plan.specific_rows, start=1):
        info = members[row.relative_path]
        destination = contained_output_path(root_path, row.relative_path)
        was_reused = extract_zip_member_no_replace(images, info, destination)
        reused += int(was_reused)
        extracted += int(not was_reused)
        with Image.open(destination) as image:
            image.load()
            if image.format != "PNG" or image.width <= 0 or image.height <= 0:
                raise RuntimeError(f"Invalid USGS publisher crop: {row.relative_path}")
            phash = perceptual_dhash(image)
            width, height = image.size
        digest = sha256_file(destination)
        image_uri = f"{USGS_IMAGES_URL}#member={row.relative_path}"
        samples.append(
            BirdSample(
                dataset_id=USGS_AERIAL_AVIAN_DATASET_ID,
                source_sample_id=row.source_sample_id,
                source_split=row.source_split,
                relative_path=row.relative_path,
                image_uri=image_uri,
                group_id=f"usgs-aerial-source-frame:{row.source_frame_id}",
                raw_label=row.raw_label,
                taxon_id=str(row.taxon_id),
                sha256=digest,
                phash=phash,
                license=USGS_AERIAL_AVIAN_LICENSE,
                author=USGS_AERIAL_AVIAN_AUTHOR,
                source=USGS_AERIAL_AVIAN_SOURCE_URL,
            )
        )
        provenance.append(
            {
                "archive_member_crc32": f"{info.CRC:08x}",
                "archive_member_size": info.file_size,
                "archive_md5": archive_audit.images_md5,
                "bbox_official": None,
                "category_id_one_based": row.category_id,
                "crop_height": height,
                "crop_phash": phash,
                "crop_provenance": "publisher-provided crop; no source bbox released",
                "crop_sha256": digest,
                "crop_width": width,
                "dataset_id": USGS_AERIAL_AVIAN_DATASET_ID,
                "encoded_label_zero_based": row.encoded_label,
                "image_uri": image_uri,
                "local_crop_applied": False,
                "official_relative_path": row.relative_path,
                "publisher_path_category_id": row.publisher_path_category_id,
                "raw_label": row.raw_label,
                "source_frame_id_derived_from_filename": row.source_frame_id,
                "source_sample_id": row.source_sample_id,
                "source_split": row.source_split,
                "taxon_id": row.taxon_id,
            }
        )
        if progress is not None:
            progress(completed, total, row.source_sample_id)
    samples.sort(key=lambda sample: sample.source_sample_id)
    provenance.sort(key=lambda value: value["source_sample_id"])
    validation = validate_manifest(
        root_path,
        samples,
        plan.taxa,
        dataset_id=USGS_AERIAL_AVIAN_DATASET_ID,
        verify_images=True,
    )
    source = {
        "annotations_archive": {
            "md5": plan.annotations_md5,
            "official_url": USGS_ANNOTATIONS_URL,
            "size_bytes": annotations.stat().st_size,
        },
        "birdnet_csv_sha256": plan.birdnet_csv_sha256,
        "citation": (
            "Miao et al. (2023), Images and annotations to automate the "
            "classification of avian species, U.S. Geological Survey data release"
        ),
        "coarse_categories_omitted": {
            _EXPECTED_LABELMAP[key]: plan.category_counts[key]
            for key in sorted(_COARSE_CATEGORY_IDS)
        },
        "dataset_id": USGS_AERIAL_AVIAN_DATASET_ID,
        "doi": USGS_AERIAL_AVIAN_DOI,
        "images_archive": {
            "md5": archive_audit.images_md5,
            "official_url": USGS_IMAGES_URL,
            "size_bytes": images.stat().st_size,
        },
        "license": USGS_AERIAL_AVIAN_LICENSE,
        "license_url": USGS_AERIAL_AVIAN_LICENSE_URL,
        "manifest_fingerprint": validation.fingerprint,
        "publisher_crop_policy": (
            "The release contains pre-cropped bird images and no source bounding boxes. "
            "TTvr copies accepted PNG members byte-for-byte and performs no local crop."
        ),
        "sample_count": len(samples),
        "source_page": USGS_AERIAL_AVIAN_SOURCE_URL,
        "source_version": USGS_AERIAL_AVIAN_VERSION,
        "split_counts": validation.split_counts,
        "taxon_count": len(plan.taxa),
    }
    publish_manifest_bundle(
        root_path,
        jsonl_files={
            "taxa.jsonl": (asdict(taxon) for taxon in plan.taxa),
            "samples.jsonl": (asdict(sample) for sample in samples),
            "crop_provenance.jsonl": provenance,
        },
        json_files={"source.json": source},
    )
    return UsgsAerialPreparation(
        annotations_md5=plan.annotations_md5,
        images_md5=archive_audit.images_md5,
        birdnet_csv_sha256=plan.birdnet_csv_sha256,
        publisher_crop_count=len(samples),
        extracted_crops=extracted,
        reused_crops=reused,
        omitted_coarse_crops=len(plan.all_rows) - len(plan.specific_rows),
        taxon_count=len(plan.taxa),
        source_group_count=len({row.source_frame_id for row in plan.specific_rows}),
        split_counts=validation.split_counts,
        manifest_fingerprint=validation.fingerprint,
    )
