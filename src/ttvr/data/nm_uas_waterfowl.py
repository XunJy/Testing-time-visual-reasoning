"""Audited expert-consensus crops from NM UAS Waterfowl on LILA BC.

Only the expert consensus COCO annotations are accepted.  ``Other`` and
``Teal`` are coarse labels and are omitted; all seven species-level categories
(including Mallard and Gadwall) remain in the canonical source manifest.
Experiment-specific CUB exclusions are reported separately and never baked
into preparation.

The LILA landing page declares CC BY-NC 4.0 while the embedded COCO ``licenses``
object still says CC BY-NC 2.0.  Both official statements are version-locked
and written to provenance instead of silently hiding the discrepancy.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import tempfile
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from PIL import Image

from .bird_crops import (
    DEFAULT_CONTEXT_SCALE,
    SQUARE_CONTEXT_RECIPE_ID,
    clip_bbox_xyxy_to_image,
    coco_bbox_to_xyxy,
    md5_file,
    save_crop_png_atomic,
    square_context_crop,
    validate_relative_posix_path,
)
from .bird_manifest import (
    BirdSample,
    BirdTaxon,
    perceptual_dhash,
    sha256_file,
    validate_manifest,
)
from .bird_source_archive import (
    atomic_publish_file_no_replace,
    contained_output_path,
    extract_zip_member_no_replace,
    publish_manifest_bundle,
    stream_download_verified,
    validated_zip_members,
)
from .birdnet_lock import BirdNetTaxonLock, resolve_locked_birdnet_taxa

NM_UAS_WATERFOWL_DATASET_ID = "nm-uas-waterfowl-expert-consensus-v1"
NM_UAS_WATERFOWL_SOURCE_VERSION = "archive-2024-02-20_expert-consensus-2023-03-31"
NM_UAS_WATERFOWL_SOURCE_URL = (
    "https://lila.science/datasets/"
    "uas-imagery-of-migratory-waterfowl-at-new-mexico-wildlife-refuges/"
)
NM_UAS_WATERFOWL_ARCHIVE_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "uas-imagery-of-migratory-waterfowl/"
    "uas-imagery-of-migratory-waterfowl.20240220.zip"
)
NM_UAS_WATERFOWL_ARCHIVE_MD5 = "4364bb40048efb4c0844324873194b9b"
NM_UAS_WATERFOWL_ARCHIVE_SIZE = 337_774_436
NM_UAS_WATERFOWL_LICENSE = "CC BY-NC 4.0"
NM_UAS_WATERFOWL_LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
NM_UAS_WATERFOWL_EMBEDDED_LICENSE = "Attribution-NonCommercial 2.0 Generic (CC BY-NC 2.0)"
NM_UAS_WATERFOWL_EMBEDDED_LICENSE_URL = "https://creativecommons.org/licenses/by-nc/2.0/"
NM_UAS_WATERFOWL_AUTHOR = (
    "Converse, Lippitt, Sesnie, Harris, Butler, and Stewart; Drones for Ducks contributors"
)
NM_UAS_WATERFOWL_CITATION = (
    "Converse RC, Lippitt CD, Sesnie SE, Harris GM, Butler MG, Stewart DR. "
    "Observer variability in manual-visual interpretation of UAS imagery of wildlife, "
    "with insights for deep learning applications."
)
NM_UAS_BIRDNET_CSV_SHA256 = "37b4015719c0c9e014d3c994dd188904457ff74edc3e3002a01b57fa9830d426"

_GCS_HOST = "storage.googleapis.com"
_ARCHIVE_PREFIX = "uas-imagery-of-migratory-waterfowl/"
_EXPERT_PREFIX = f"{_ARCHIVE_PREFIX}experts/"
_EXPERT_IMAGE_PREFIX = f"{_EXPERT_PREFIX}images/"
_CROWD_IMAGE_PREFIX = f"{_ARCHIVE_PREFIX}crowdsourced/images/"
NM_UAS_EXPERT_REFINED_MEMBER = f"{_EXPERT_PREFIX}20230331_dronesforducks_expert_refined.json"
_OTHER_METADATA_MEMBERS = {
    f"{_ARCHIVE_PREFIX}crowdsourced/20240209_dronesforducks_zooniverse_raw.json",
    f"{_ARCHIVE_PREFIX}crowdsourced/20240220_dronesforducks_zooniverse_refined.json",
    f"{_EXPERT_PREFIX}20230331_dronesforducks_raw_experts.json",
}
NM_UAS_EXPERT_REFINED_SIZE = 224_053
NM_UAS_EXPERT_REFINED_COMPRESSED_SIZE = 29_669
NM_UAS_EXPERT_REFINED_CRC32 = 0xEAF05431
NM_UAS_EXPERT_REFINED_SHA256 = "114018d60f88adc4285e92349eac50ae943c7735370d8b5fec76f452387928f6"
NM_UAS_EXPERT_REFINED_LOCAL_HEADER_OFFSET = 212_867_297
NM_UAS_EXPERT_REFINED_RANGE_SIZE = 29_785
_EXPECTED_ARCHIVE_COUNTS = {
    "entries": 377,
    "directories": 5,
    "files": 372,
    "expert_images": 12,
    "crowdsourced_images": 356,
}
_EXPECTED_CATEGORY_NAMES = {
    1: "Canadian Goose",
    2: "Sandhill Crane",
    3: "Mallard",
    4: "Northern Pintail",
    5: "American Wigeon",
    6: "Other",
    7: "Teal",
    8: "Gadwall",
    9: "Northern Shoveler",
}
_EXPECTED_CATEGORY_COUNTS = {
    1: 140,
    2: 52,
    3: 1_688,
    4: 262,
    5: 22,
    6: 70,
    7: 2,
    8: 5,
    9: 2,
}
_COARSE_CATEGORY_IDS = frozenset({6, 7})
_SPECIFIC_CATEGORY_IDS = frozenset(set(_EXPECTED_CATEGORY_NAMES) - _COARSE_CATEGORY_IDS)
_EXPECTED_COUNTS = {
    "images": 12,
    "categories": 9,
    "annotations": 2_243,
    "specific_crops": 2_171,
    "coarse_crops": 72,
    "taxa": 7,
    "clipped_bboxes": 0,
}
_SAFE_ID = re.compile(r"[1-9][0-9]*\Z")
_IMAGE_FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\.JPG\Z")
_ARCHIVE_EXPERT_IMAGE_RE = re.compile(
    re.escape(_EXPERT_IMAGE_PREFIX) + r"[A-Za-z0-9][A-Za-z0-9_-]*\.jpg\Z"
)
_ARCHIVE_CROWD_IMAGE_RE = re.compile(
    re.escape(_CROWD_IMAGE_PREFIX) + r"[A-Za-z0-9][A-Za-z0-9_-]*\.png\Z"
)

NM_UAS_WATERFOWL_TAXON_LOCKS: tuple[BirdNetTaxonLock, ...] = (
    # The publisher's "Canadian Goose" is a reviewed common-name typo.
    BirdNetTaxonLock("Canadian Goose", "BN01898", "Branta canadensis", "Canada Goose"),
    BirdNetTaxonLock("Sandhill Crane", "BN00949", "Antigone canadensis", "Sandhill Crane"),
    BirdNetTaxonLock("Mallard", "BN00713", "Anas platyrhynchos", "Mallard"),
    BirdNetTaxonLock("Northern Pintail", "BN00691", "Anas acuta", "Northern Pintail"),
    BirdNetTaxonLock("American Wigeon", "BN08400", "Mareca americana", "American Wigeon"),
    BirdNetTaxonLock("Gadwall", "BN08405", "Mareca strepera", "Gadwall"),
    BirdNetTaxonLock("Northern Shoveler", "BN14030", "Spatula clypeata", "Northern Shoveler"),
)


@dataclass(frozen=True, slots=True)
class NmUasImagePlan:
    image_id: int
    file_name: str
    archive_member: str
    width: int
    height: int
    date_captured: str


@dataclass(frozen=True, slots=True)
class NmUasCropPlan:
    annotation_id: int
    image_id: int
    category_id: int
    raw_label: str
    taxon_id: str
    bbox_xywh: tuple[float, float, float, float]
    bbox_xyxy_original: tuple[float, float, float, float]
    bbox_xyxy_effective: tuple[float, float, float, float]
    bbox_was_clipped: bool


@dataclass(frozen=True, slots=True)
class NmUasMetadataPlan:
    metadata_path: Path
    metadata_sha256: str
    birdnet_csv_sha256: str
    images: tuple[NmUasImagePlan, ...]
    crops: tuple[NmUasCropPlan, ...]
    taxa: tuple[BirdTaxon, ...]
    category_counts: dict[int, int]
    coarse_crop_count: int
    total_annotation_count: int
    clipped_bbox_count: int

    @property
    def image_by_id(self) -> dict[int, NmUasImagePlan]:
        return {image.image_id: image for image in self.images}


@dataclass(frozen=True, slots=True)
class NmUasArchiveAudit:
    archive_md5: str
    entry_count: int
    file_count: int
    expert_image_count: int
    crowdsourced_image_count: int
    embedded_metadata_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NmUasPreparation:
    archive_md5: str
    metadata_sha256: str
    birdnet_csv_sha256: str
    source_image_count: int
    extracted_source_images: int
    reused_source_images: int
    crop_count: int
    written_crops: int
    reused_crops: int
    coarse_crops_omitted: int
    taxon_count: int
    clipped_bbox_count: int
    manifest_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NmUasStrictCubAudit:
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


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"NM UAS {field} must be a positive integer")
    return value


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Malformed NM UAS expert consensus JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError("NM UAS expert consensus metadata must be an object")
    required = {"info", "images", "categories", "annotations", "licenses"}
    if set(value) != required:
        raise RuntimeError(
            f"Unexpected NM UAS metadata keys: {sorted(value)}; expected {sorted(required)}"
        )
    if (
        not isinstance(value["info"], dict)
        or not isinstance(value["licenses"], dict)
        or any(not isinstance(value[key], list) for key in ("images", "categories", "annotations"))
    ):
        raise RuntimeError("NM UAS metadata has an invalid top-level schema")
    return value


def plan_nm_uas_waterfowl(
    metadata_path: Path | str,
    *,
    birdnet_csv_path: Path | str,
    require_official_lock: bool = True,
) -> NmUasMetadataPlan:
    """Validate consensus schema, boxes, and taxonomy before source images are used."""

    metadata = Path(metadata_path).expanduser()
    if metadata.is_symlink() or not metadata.is_file():
        raise FileNotFoundError(f"Missing regular NM UAS expert metadata: {metadata}")
    metadata_sha256 = sha256_file(metadata)
    if require_official_lock and (
        metadata_sha256 != NM_UAS_EXPERT_REFINED_SHA256
        or metadata.stat().st_size != NM_UAS_EXPERT_REFINED_SIZE
    ):
        raise RuntimeError(
            "NM UAS expert metadata differs from the official archive member: "
            f"size={metadata.stat().st_size}, sha256={metadata_sha256}"
        )
    payload = _load_metadata(metadata)
    if require_official_lock:
        info = payload["info"]
        expected_info = {
            "year": 2023,
            "version": "1.0",
            "contributor": (
                "Rowan Converse, Center for the Advancement of Spatial Informatics Education"
            ),
            "url": "http://aspire.unm.edu/projects/project/ducks-and-drones.html",
            "date created": "2023-03-31",
        }
        # The description is long prose; lock its digest while checking every other field exactly.
        description = info.get("description")
        if not isinstance(description, str) or not description.startswith(
            "This dataset is composed of consensus annotations"
        ):
            raise RuntimeError("NM UAS info.description changed")
        remaining_info = {key: value for key, value in info.items() if key != "description"}
        # The official file says "and Education"; retain an exact check below without
        # normalizing the typo-prone institutional name.
        expected_info["contributor"] = (
            "Rowan Converse, Center for the Advancement of Spatial Informatics Research "
            "and Education"
        )
        if remaining_info != expected_info:
            raise RuntimeError(f"NM UAS info metadata changed: {remaining_info}")
        expected_license = {
            "id": 1,
            "url": NM_UAS_WATERFOWL_EMBEDDED_LICENSE_URL,
            "name": NM_UAS_WATERFOWL_EMBEDDED_LICENSE,
        }
        if payload["licenses"] != expected_license:
            raise RuntimeError(f"NM UAS embedded COCO license changed: {payload['licenses']}")

    images_by_id: dict[int, NmUasImagePlan] = {}
    file_names: set[str] = set()
    for row in payload["images"]:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "file_name",
            "license",
            "width",
            "height",
            "date_captured",
        }:
            raise RuntimeError("Malformed NM UAS image row")
        image_id = _positive_int(row["id"], field="image id")
        file_name = str(row["file_name"])
        validate_relative_posix_path(file_name)
        if "/" in file_name or not _IMAGE_FILE_RE.fullmatch(file_name):
            raise RuntimeError(f"Unexpected NM UAS expert image name: {file_name}")
        if row["license"] != 1:
            raise RuntimeError(f"Unexpected NM UAS image license id: {image_id}")
        width = _positive_int(row["width"], field="image width")
        height = _positive_int(row["height"], field="image height")
        date_captured = str(row["date_captured"])
        if not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}", date_captured
        ):
            raise RuntimeError(f"Unexpected NM UAS capture date: {date_captured}")
        if image_id in images_by_id or file_name.casefold() in file_names:
            raise RuntimeError(f"Duplicate NM UAS image identity: {image_id}/{file_name}")
        file_names.add(file_name.casefold())
        archive_member = f"{_EXPERT_IMAGE_PREFIX}{PurePosixPath(file_name).stem}.jpg"
        images_by_id[image_id] = NmUasImagePlan(
            image_id=image_id,
            file_name=file_name,
            archive_member=archive_member,
            width=width,
            height=height,
            date_captured=date_captured,
        )

    categories: dict[int, str] = {}
    for row in payload["categories"]:
        if not isinstance(row, dict) or set(row) - {"id", "name", "supercategory"}:
            raise RuntimeError("Malformed NM UAS category row")
        if set(row) < {"id", "name"}:
            raise RuntimeError("NM UAS category lacks id or name")
        category_id = _positive_int(row["id"], field="category id")
        name = row["name"]
        if not isinstance(name, str) or not name.strip() or category_id in categories:
            raise RuntimeError(f"Invalid or duplicate NM UAS category: {row}")
        supercategory = row.get("supercategory")
        if supercategory is not None and (
            not isinstance(supercategory, str) or not supercategory.strip()
        ):
            raise RuntimeError(f"Invalid NM UAS supercategory: {row}")
        categories[category_id] = name
    if require_official_lock and categories != _EXPECTED_CATEGORY_NAMES:
        raise RuntimeError(f"NM UAS category names changed: {categories}")
    if set(categories) != set(_EXPECTED_CATEGORY_NAMES):
        raise RuntimeError(f"Unexpected NM UAS category ids: {sorted(categories)}")

    taxa_by_label, birdnet_digest = resolve_locked_birdnet_taxa(
        birdnet_csv_path,
        NM_UAS_WATERFOWL_TAXON_LOCKS,
        expected_sha256=NM_UAS_BIRDNET_CSV_SHA256,
        require_official_lock=require_official_lock,
    )
    annotation_ids: set[int] = set()
    category_counts: Counter[int] = Counter()
    crops: list[NmUasCropPlan] = []
    clipped_bbox_count = 0
    for row in payload["annotations"]:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "image_id",
            "category_id",
            "bbox",
            "iscrowd",
        }:
            raise RuntimeError("Malformed NM UAS annotation row")
        annotation_id = _positive_int(row["id"], field="annotation id")
        if annotation_id in annotation_ids:
            raise RuntimeError(f"Duplicate NM UAS annotation id: {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = _positive_int(row["image_id"], field="annotation image id")
        image = images_by_id.get(image_id)
        if image is None:
            raise RuntimeError(f"NM UAS annotation references unknown image: {image_id}")
        category_id = _positive_int(row["category_id"], field="annotation category id")
        if category_id not in categories:
            raise RuntimeError(f"NM UAS annotation references unknown category: {category_id}")
        if row["iscrowd"] != 0:
            raise RuntimeError(f"NM UAS consensus annotation is marked crowd: {annotation_id}")
        bbox = row["bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(value, bool) for value in bbox)
        ):
            raise RuntimeError(f"Malformed NM UAS bbox: {annotation_id}")
        try:
            bbox_xywh = tuple(float(value) for value in bbox)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Non-numeric NM UAS bbox: {annotation_id}") from error
        bbox_original = coco_bbox_to_xyxy(bbox_xywh)
        bbox_effective, was_clipped = clip_bbox_xyxy_to_image(
            bbox_original,
            (image.width, image.height),
        )
        clipped_bbox_count += int(was_clipped)
        category_counts[category_id] += 1
        if category_id in _COARSE_CATEGORY_IDS:
            continue
        raw_label = categories[category_id]
        taxon = taxa_by_label.get(raw_label)
        if taxon is None:
            raise RuntimeError(f"Specific NM UAS label lacks a reviewed taxon: {raw_label}")
        crops.append(
            NmUasCropPlan(
                annotation_id=annotation_id,
                image_id=image_id,
                category_id=category_id,
                raw_label=raw_label,
                taxon_id=taxon.taxon_id,
                bbox_xywh=bbox_xywh,
                bbox_xyxy_original=bbox_original,
                bbox_xyxy_effective=bbox_effective,
                bbox_was_clipped=was_clipped,
            )
        )
    taxa = tuple(sorted(taxa_by_label.values(), key=lambda taxon: taxon.taxon_id))
    observed_counts = {
        "images": len(images_by_id),
        "categories": len(categories),
        "annotations": len(annotation_ids),
        "specific_crops": len(crops),
        "coarse_crops": len(annotation_ids) - len(crops),
        "taxa": len(taxa),
        "clipped_bboxes": clipped_bbox_count,
    }
    if require_official_lock and (
        observed_counts != _EXPECTED_COUNTS
        or dict(sorted(category_counts.items())) != _EXPECTED_CATEGORY_COUNTS
    ):
        raise RuntimeError(
            f"Unexpected NM UAS official counts: {observed_counts}, "
            f"categories={dict(category_counts)}"
        )
    del payload
    return NmUasMetadataPlan(
        metadata_path=metadata.resolve(),
        metadata_sha256=metadata_sha256,
        birdnet_csv_sha256=birdnet_digest,
        images=tuple(sorted(images_by_id.values(), key=lambda image: image.image_id)),
        crops=tuple(sorted(crops, key=lambda crop: crop.annotation_id)),
        taxa=taxa,
        category_counts=dict(sorted(category_counts.items())),
        coarse_crop_count=len(annotation_ids) - len(crops),
        total_annotation_count=len(annotation_ids),
        clipped_bbox_count=clipped_bbox_count,
    )


def _decode_locked_metadata_range(value: bytes) -> bytes:
    if len(value) != NM_UAS_EXPERT_REFINED_RANGE_SIZE:
        raise RuntimeError(
            f"NM UAS metadata range length changed: {len(value)}; "
            f"expected {NM_UAS_EXPERT_REFINED_RANGE_SIZE}"
        )
    if len(value) < 30:
        raise RuntimeError("NM UAS metadata range lacks a ZIP local header")
    header = struct.unpack("<IHHHHHIIIHH", value[:30])
    (
        signature,
        _version,
        flags,
        compression,
        _mod_time,
        _mod_date,
        crc32_value,
        compressed_size,
        file_size,
        filename_length,
        extra_length,
    ) = header
    if signature != 0x04034B50 or flags != 0 or compression != zipfile.ZIP_DEFLATED:
        raise RuntimeError("NM UAS metadata ZIP local header changed")
    filename_end = 30 + filename_length
    data_start = filename_end + extra_length
    try:
        filename = value[30:filename_end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("NM UAS metadata member name is not UTF-8") from error
    expected = (
        NM_UAS_EXPERT_REFINED_MEMBER,
        NM_UAS_EXPERT_REFINED_CRC32,
        NM_UAS_EXPERT_REFINED_COMPRESSED_SIZE,
        NM_UAS_EXPERT_REFINED_SIZE,
    )
    observed = (filename, crc32_value, compressed_size, file_size)
    if observed != expected or data_start + compressed_size != len(value):
        raise RuntimeError(
            f"NM UAS metadata ZIP member changed: expected {expected}, found {observed}"
        )
    compressed = value[data_start:]
    try:
        result = zlib.decompress(compressed, -zlib.MAX_WBITS)
    except zlib.error as error:
        raise RuntimeError("NM UAS metadata ZIP member is not valid deflate data") from error
    if (
        len(result) != NM_UAS_EXPERT_REFINED_SIZE
        or zlib.crc32(result) & 0xFFFFFFFF != NM_UAS_EXPERT_REFINED_CRC32
        or hashlib_sha256(result) != NM_UAS_EXPERT_REFINED_SHA256
    ):
        raise RuntimeError("NM UAS ranged expert metadata checksum mismatch")
    return result


def hashlib_sha256(value: bytes) -> str:
    # Local helper keeps the source-data API independent of file materialization.
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _request_locked_metadata_range(*, timeout: int = 60) -> bytes:
    parsed = urlsplit(NM_UAS_WATERFOWL_ARCHIVE_URL)
    if parsed.scheme != "https" or parsed.hostname != _GCS_HOST:
        raise ValueError("NM UAS archive URL is not the locked GCS object")
    head_request = urllib.request.Request(
        NM_UAS_WATERFOWL_ARCHIVE_URL,
        method="HEAD",
        headers={"User-Agent": "ttvr-birdmix/1.0"},
    )
    with urllib.request.urlopen(head_request, timeout=timeout) as response:
        final = urlsplit(response.geturl())
        if final.scheme != "https" or final.hostname != _GCS_HOST:
            raise RuntimeError(f"NM UAS archive redirected to: {response.geturl()}")
        try:
            content_length = int(response.headers["Content-Length"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("NM UAS archive HEAD lacks a valid Content-Length") from error
        etag = str(response.headers.get("ETag", "")).strip('"')
        accepts_ranges = str(response.headers.get("Accept-Ranges", "")).casefold()
    if (
        content_length != NM_UAS_WATERFOWL_ARCHIVE_SIZE
        or etag != NM_UAS_WATERFOWL_ARCHIVE_MD5
        or accepts_ranges != "bytes"
    ):
        raise RuntimeError(
            "NM UAS GCS object metadata changed: "
            f"size={content_length}, etag={etag}, accept-ranges={accepts_ranges}"
        )
    start = NM_UAS_EXPERT_REFINED_LOCAL_HEADER_OFFSET
    end = start + NM_UAS_EXPERT_REFINED_RANGE_SIZE - 1
    range_request = urllib.request.Request(
        NM_UAS_WATERFOWL_ARCHIVE_URL,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "ttvr-birdmix/1.0",
        },
    )
    with urllib.request.urlopen(range_request, timeout=timeout) as response:
        final = urlsplit(response.geturl())
        if final.scheme != "https" or final.hostname != _GCS_HOST:
            raise RuntimeError(f"NM UAS range redirected to: {response.geturl()}")
        status = getattr(response, "status", response.getcode())
        content_range = str(response.headers.get("Content-Range", ""))
        payload = response.read(NM_UAS_EXPERT_REFINED_RANGE_SIZE + 1)
    expected_range = f"bytes {start}-{end}/{NM_UAS_WATERFOWL_ARCHIVE_SIZE}"
    if status != 206 or content_range != expected_range:
        raise RuntimeError(
            f"NM UAS server did not honor the locked byte range: "
            f"status={status}, Content-Range={content_range!r}"
        )
    return _decode_locked_metadata_range(payload)


def download_nm_uas_metadata(
    root: Path | str = Path("data/nm_uas_waterfowl"),
) -> Path:
    """Fetch only the 224-KB expert consensus member using a locked HTTP range."""

    root_path = Path(root).expanduser().resolve()
    destination = contained_output_path(
        root_path,
        "sources/20230331_dronesforducks_expert_refined.json",
    )
    if destination.is_symlink():
        raise ValueError(f"NM UAS metadata destination must not be a symlink: {destination}")
    if destination.exists():
        if (
            not destination.is_file()
            or destination.stat().st_size != NM_UAS_EXPERT_REFINED_SIZE
            or sha256_file(destination) != NM_UAS_EXPERT_REFINED_SHA256
        ):
            raise RuntimeError(f"Existing NM UAS expert metadata mismatch: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _request_locked_metadata_range()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.part-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            atomic_publish_file_no_replace(temporary, destination)
        except FileExistsError as error:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.stat().st_size != NM_UAS_EXPERT_REFINED_SIZE
                or sha256_file(destination) != NM_UAS_EXPERT_REFINED_SHA256
            ):
                raise RuntimeError(
                    "Concurrent NM UAS expert metadata mismatch"
                ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _ensure_archive(root: Path) -> Path:
    destination = contained_output_path(root, "sources/uas-imagery-of-migratory-waterfowl.zip")
    stream_download_verified(
        NM_UAS_WATERFOWL_ARCHIVE_URL,
        destination,
        expected_md5=NM_UAS_WATERFOWL_ARCHIVE_MD5,
        expected_size=NM_UAS_WATERFOWL_ARCHIVE_SIZE,
        allowed_hosts={_GCS_HOST},
    )
    return destination


def audit_nm_uas_archive(
    plan: NmUasMetadataPlan,
    archive_path: Path | str,
    *,
    require_official_lock: bool = True,
) -> NmUasArchiveAudit:
    """Verify archive checksum, complete member schema, and embedded metadata identity."""

    archive = Path(archive_path).expanduser()
    if archive.is_symlink() or not archive.is_file():
        raise FileNotFoundError(f"Missing regular NM UAS archive: {archive}")
    archive_md5 = md5_file(archive)
    if require_official_lock and (
        archive_md5 != NM_UAS_WATERFOWL_ARCHIVE_MD5
        or archive.stat().st_size != NM_UAS_WATERFOWL_ARCHIVE_SIZE
    ):
        raise RuntimeError(
            "NM UAS archive differs from the official GCS object: "
            f"size={archive.stat().st_size}, md5={archive_md5}"
        )
    members = validated_zip_members(archive)
    files = {name for name, info in members.items() if not info.is_dir()}
    expert_images = {name for name in files if name.startswith(_EXPERT_IMAGE_PREFIX)}
    crowd_images = {name for name in files if name.startswith(_CROWD_IMAGE_PREFIX)}
    metadata_files = files - expert_images - crowd_images
    expected_expert = {image.archive_member for image in plan.images}
    if expert_images != expected_expert:
        raise RuntimeError(
            "NM UAS expert image members do not match consensus metadata: "
            f"missing={sorted(expected_expert - expert_images)}, "
            f"extra={sorted(expert_images - expected_expert)}"
        )
    if any(not _ARCHIVE_EXPERT_IMAGE_RE.fullmatch(name) for name in expert_images):
        raise RuntimeError("NM UAS archive has an unexpected expert image path")
    if any(not _ARCHIVE_CROWD_IMAGE_RE.fullmatch(name) for name in crowd_images):
        raise RuntimeError("NM UAS archive has an unexpected crowdsourced image path")
    expected_metadata = _OTHER_METADATA_MEMBERS | {NM_UAS_EXPERT_REFINED_MEMBER}
    if metadata_files != expected_metadata:
        raise RuntimeError(f"NM UAS archive metadata members changed: {sorted(metadata_files)}")
    with zipfile.ZipFile(archive) as bundle:
        embedded_metadata = bundle.read(NM_UAS_EXPERT_REFINED_MEMBER)
    embedded_sha256 = hashlib_sha256(embedded_metadata)
    if embedded_sha256 != plan.metadata_sha256:
        raise RuntimeError(
            "NM UAS ranged/local expert metadata differs from the full archive member"
        )
    observed_counts = {
        "entries": len(members),
        "directories": sum(info.is_dir() for info in members.values()),
        "files": len(files),
        "expert_images": len(expert_images),
        "crowdsourced_images": len(crowd_images),
    }
    if require_official_lock and observed_counts != _EXPECTED_ARCHIVE_COUNTS:
        raise RuntimeError(f"Unexpected NM UAS archive counts: {observed_counts}")
    return NmUasArchiveAudit(
        archive_md5=archive_md5,
        entry_count=len(members),
        file_count=len(files),
        expert_image_count=len(expert_images),
        crowdsourced_image_count=len(crowd_images),
        embedded_metadata_sha256=embedded_sha256,
    )


def audit_nm_uas_strict_cub(
    rows: Sequence[NmUasCropPlan | BirdSample],
    excluded_birdnet_ids: Iterable[str],
) -> NmUasStrictCubAudit:
    """Report target-taxon removal without changing canonical consensus rows."""

    if not rows:
        raise ValueError("NM UAS strict-CUB audit rows must not be empty")
    excluded = {
        value if value.startswith("birdnet:") else f"birdnet:{value}"
        for raw in excluded_birdnet_ids
        if (value := str(raw).strip())
    }
    counts = Counter(row.taxon_id for row in rows)
    excluded_counts = {key: value for key, value in counts.items() if key in excluded}
    return NmUasStrictCubAudit(
        total_crops=len(rows),
        total_taxa=len(counts),
        excluded_crops=sum(excluded_counts.values()),
        excluded_taxa=len(excluded_counts),
        retained_crops=len(rows) - sum(excluded_counts.values()),
        retained_taxa=len(counts) - len(excluded_counts),
        excluded_counts_by_taxon=tuple(sorted(excluded_counts.items())),
    )


def _reuse_or_write_crop(image: Image.Image, destination: Path) -> tuple[str, str, bool]:
    if destination.is_symlink():
        raise ValueError(f"NM UAS crop path must not be a symlink: {destination}")
    if destination.is_file():
        with Image.open(destination) as existing_image:
            existing_image.load()
            existing = existing_image.convert("RGB")
        expected = image.convert("RGB")
        if existing.size != expected.size or existing.tobytes() != expected.tobytes():
            raise RuntimeError(f"Existing NM UAS crop differs from recipe: {destination}")
        return sha256_file(destination), perceptual_dhash(existing), True
    try:
        digest, phash = save_crop_png_atomic(image, destination)
        return digest, phash, False
    except FileExistsError:
        return _reuse_or_write_crop(image, destination)


def prepare_nm_uas_waterfowl(
    root: Path | str = Path("data/nm_uas_waterfowl"),
    *,
    birdnet_csv_path: Path | str,
    metadata_path: Path | str | None = None,
    archive_path: Path | str | None = None,
    context_scale: float = DEFAULT_CONTEXT_SCALE,
    keep_source_images: bool = False,
    require_official_lock: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> NmUasPreparation:
    """Extract expert images and create deterministic crops for all specific boxes."""

    if (
        not isinstance(context_scale, (int, float))
        or isinstance(context_scale, bool)
        or not math.isfinite(context_scale)
        or context_scale < 1.0
    ):
        raise ValueError("context_scale must be finite and at least 1.0")
    root_path = Path(root).expanduser().resolve()
    manifests = root_path / "manifests"
    if manifests.exists() or manifests.is_symlink():
        raise FileExistsError(f"Refusing to replace manifest directory: {manifests}")
    if not require_official_lock and (metadata_path is None or archive_path is None):
        raise ValueError("non-official preparation requires explicit metadata and archive paths")
    metadata = (
        download_nm_uas_metadata(root_path)
        if metadata_path is None
        else Path(metadata_path).expanduser()
    )
    archive = (
        _ensure_archive(root_path) if archive_path is None else Path(archive_path).expanduser()
    )
    plan = plan_nm_uas_waterfowl(
        metadata,
        birdnet_csv_path=birdnet_csv_path,
        require_official_lock=require_official_lock,
    )
    archive_audit = audit_nm_uas_archive(
        plan,
        archive,
        require_official_lock=require_official_lock,
    )
    members = validated_zip_members(archive)
    crops_by_image: dict[int, list[NmUasCropPlan]] = defaultdict(list)
    for crop in plan.crops:
        crops_by_image[crop.image_id].append(crop)
    samples: list[BirdSample] = []
    provenance: list[dict[str, Any]] = []
    source_images: list[dict[str, Any]] = []
    extracted_sources = 0
    reused_sources = 0
    written_crops = 0
    reused_crops = 0
    completed_crops = 0
    total_crops = len(plan.crops)
    for image_plan in plan.images:
        source_relative = f"source_images/{PurePosixPath(image_plan.archive_member).name}"
        source_path = contained_output_path(root_path, source_relative)
        was_reused = extract_zip_member_no_replace(
            archive,
            members[image_plan.archive_member],
            source_path,
        )
        reused_sources += int(was_reused)
        extracted_sources += int(not was_reused)
        source_sha256 = sha256_file(source_path)
        with Image.open(source_path) as raw:
            raw.load()
            if raw.size != (image_plan.width, image_plan.height):
                raise RuntimeError(
                    f"NM UAS source dimensions changed for image {image_plan.image_id}: "
                    f"metadata={(image_plan.width, image_plan.height)}, image={raw.size}"
                )
            source_image = raw.convert("RGB")
        image_crops = sorted(
            crops_by_image.get(image_plan.image_id, []),
            key=lambda crop: crop.annotation_id,
        )
        for crop_plan in image_crops:
            crop, geometry = square_context_crop(
                source_image,
                crop_plan.bbox_xyxy_effective,
                context_scale=context_scale,
            )
            relative_path = f"crops/{crop_plan.annotation_id}.png"
            destination = contained_output_path(root_path, relative_path)
            crop_sha256, crop_phash, crop_was_reused = _reuse_or_write_crop(
                crop,
                destination,
            )
            reused_crops += int(crop_was_reused)
            written_crops += int(not crop_was_reused)
            image_uri = f"{NM_UAS_WATERFOWL_ARCHIVE_URL}#member={image_plan.archive_member}"
            samples.append(
                BirdSample(
                    dataset_id=NM_UAS_WATERFOWL_DATASET_ID,
                    source_sample_id=f"expert-consensus:{crop_plan.annotation_id}",
                    source_split="train",
                    relative_path=relative_path,
                    image_uri=image_uri,
                    group_id=f"nm-uas-source-image:{image_plan.image_id}",
                    raw_label=crop_plan.raw_label,
                    taxon_id=crop_plan.taxon_id,
                    sha256=crop_sha256,
                    phash=crop_phash,
                    license=NM_UAS_WATERFOWL_LICENSE,
                    author=NM_UAS_WATERFOWL_AUTHOR,
                    source=NM_UAS_WATERFOWL_SOURCE_URL,
                )
            )
            provenance.append(
                {
                    "annotation_id": crop_plan.annotation_id,
                    "archive_md5": archive_audit.archive_md5,
                    "archive_source_member": image_plan.archive_member,
                    "bbox_was_clipped": crop_plan.bbox_was_clipped,
                    "bbox_xywh_official": list(crop_plan.bbox_xywh),
                    "bbox_xyxy_effective": list(crop_plan.bbox_xyxy_effective),
                    "bbox_xyxy_official": list(crop_plan.bbox_xyxy_original),
                    "category_id": crop_plan.category_id,
                    "crop_geometry": geometry.to_dict(),
                    "crop_phash": crop_phash,
                    "crop_relative_path": relative_path,
                    "crop_sha256": crop_sha256,
                    "dataset_id": NM_UAS_WATERFOWL_DATASET_ID,
                    "expert_consensus_only": True,
                    "image_uri": image_uri,
                    "raw_label": crop_plan.raw_label,
                    "source_image_id": image_plan.image_id,
                    "source_image_sha256": source_sha256,
                    "taxon_id": crop_plan.taxon_id,
                }
            )
            completed_crops += 1
            if progress is not None:
                progress(
                    completed_crops,
                    total_crops,
                    f"expert-consensus:{crop_plan.annotation_id}",
                )
        source_images.append(
            {
                "archive_member": image_plan.archive_member,
                "archive_member_crc32": f"{members[image_plan.archive_member].CRC:08x}",
                "archive_md5": archive_audit.archive_md5,
                "capture_date": image_plan.date_captured,
                "crop_count": len(image_crops),
                "group_id": f"nm-uas-source-image:{image_plan.image_id}",
                "height": image_plan.height,
                "image_id": image_plan.image_id,
                "source_image_sha256": source_sha256,
                "width": image_plan.width,
            }
        )
        if not keep_source_images:
            source_path.unlink(missing_ok=True)
    if completed_crops != total_crops:
        raise RuntimeError(
            f"NM UAS crop coverage mismatch: completed {completed_crops}/{total_crops}"
        )
    samples.sort(key=lambda sample: sample.source_sample_id)
    provenance.sort(key=lambda value: value["annotation_id"])
    source_images.sort(key=lambda value: value["image_id"])
    validation = validate_manifest(
        root_path,
        samples,
        plan.taxa,
        dataset_id=NM_UAS_WATERFOWL_DATASET_ID,
        verify_images=True,
    )
    source = {
        "archive": {
            "md5": archive_audit.archive_md5,
            "official_url": NM_UAS_WATERFOWL_ARCHIVE_URL,
            "size_bytes": archive.stat().st_size,
        },
        "birdnet_csv_sha256": plan.birdnet_csv_sha256,
        "citation": NM_UAS_WATERFOWL_CITATION,
        "coarse_categories_omitted": {
            _EXPECTED_CATEGORY_NAMES[key]: plan.category_counts[key]
            for key in sorted(_COARSE_CATEGORY_IDS)
        },
        "crop_recipe": {
            "context_scale": float(context_scale),
            "recipe_id": SQUARE_CONTEXT_RECIPE_ID,
        },
        "dataset_id": NM_UAS_WATERFOWL_DATASET_ID,
        "expert_consensus_member": {
            "member_name": NM_UAS_EXPERT_REFINED_MEMBER,
            "sha256": plan.metadata_sha256,
            "size_bytes": metadata.stat().st_size,
        },
        "label_policy": "expert consensus only; Other and Teal omitted",
        "landing_page_license": NM_UAS_WATERFOWL_LICENSE,
        "landing_page_license_url": NM_UAS_WATERFOWL_LICENSE_URL,
        "license_metadata_discrepancy": (
            "The LILA landing page declares CC BY-NC 4.0, while the embedded COCO "
            "licenses object declares CC BY-NC 2.0. Both statements are retained."
        ),
        "manifest_fingerprint": validation.fingerprint,
        "sample_count": len(samples),
        "source_page": NM_UAS_WATERFOWL_SOURCE_URL,
        "source_version": NM_UAS_WATERFOWL_SOURCE_VERSION,
        "taxon_count": len(plan.taxa),
        "embedded_coco_license": NM_UAS_WATERFOWL_EMBEDDED_LICENSE,
        "embedded_coco_license_url": NM_UAS_WATERFOWL_EMBEDDED_LICENSE_URL,
    }
    publish_manifest_bundle(
        root_path,
        jsonl_files={
            "taxa.jsonl": (asdict(taxon) for taxon in plan.taxa),
            "samples.jsonl": (asdict(sample) for sample in samples),
            "crop_provenance.jsonl": provenance,
            "source_images.jsonl": source_images,
        },
        json_files={"source.json": source},
    )
    return NmUasPreparation(
        archive_md5=archive_audit.archive_md5,
        metadata_sha256=plan.metadata_sha256,
        birdnet_csv_sha256=plan.birdnet_csv_sha256,
        source_image_count=len(plan.images),
        extracted_source_images=extracted_sources,
        reused_source_images=reused_sources,
        crop_count=len(samples),
        written_crops=written_crops,
        reused_crops=reused_crops,
        coarse_crops_omitted=plan.coarse_crop_count,
        taxon_count=len(plan.taxa),
        clipped_bbox_count=plan.clipped_bbox_count,
        manifest_fingerprint=validation.fingerprint,
    )
