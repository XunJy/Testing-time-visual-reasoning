"""Audited preparation of species crops from the official Big Bird dataset.

Only the official metadata is fetched up front.  Source UAV images are then
downloaded individually *only* when they contain at least one species-level
bounding box.  The generic ``bird`` category is deliberately omitted.  Every
Google Cloud Storage object MD5 and every derived crop hash is retained in an
immutable provenance bundle.

The canonical manifest contains every resolvable source species.  CUB target
taxa are never removed here; :func:`audit_big_bird_strict_cub` reports the
effect of a caller-supplied exclusion set without changing source data.
"""

from __future__ import annotations

import base64
import binascii
import csv
import json
import math
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from PIL import Image

from .bird_crops import (
    DEFAULT_CONTEXT_SCALE,
    SQUARE_CONTEXT_RECIPE_ID,
    clip_bbox_xyxy_to_image,
    coco_bbox_to_xyxy,
    md5_base64_file,
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
    write_json,
    write_jsonl,
)
from .birdnet import (
    BIRDNET_CSV_URL,
    BIRDNET_TAXONOMY_AUTHORITY,
    BIRDNET_TAXONOMY_VERSION,
)

BIG_BIRD_DATASET_ID = "big-bird-v2026.03.21-bbox-crops"
BIG_BIRD_SOURCE_VERSION = "2026.03.21"
BIG_BIRD_SOURCE_URL = "https://lila.science/datasets/big-bird/"
BIG_BIRD_METADATA_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "big-bird/wilson-bigbird.json"
)
BIG_BIRD_METADATA_FILENAME = "wilson-bigbird.json"
BIG_BIRD_METADATA_OBSERVED_SHA256 = (
    "f18f31f76bcd4c6471d3d9a5fff0b0ef1fb56fca8a294e966fd595952594252a"
)
BIG_BIRD_BIRDNET_CSV_OBSERVED_SHA256 = (
    "37b4015719c0c9e014d3c994dd188904457ff74edc3e3002a01b57fa9830d426"
)
BIG_BIRD_LICENSE_URL = (
    "https://guides.library.uq.edu.au/research-and-teaching-staff/"
    "data-deposit-checklist/license-reuse-with-acknowledgement"
)
BIG_BIRD_LICENSE = (
    "University of Queensland Permitted Re-Use with Acknowledgement "
    f"({BIG_BIRD_LICENSE_URL})"
)
BIG_BIRD_CITATION = (
    "Wilson et al. (2026), Big Bird: A global dataset of birds in drone imagery "
    "annotated to species level, Remote Sensing in Ecology and Conservation"
)
BIG_BIRD_AUTHOR = "Wilson et al. (2026) Big Bird contributors"
BIG_BIRD_GCS_BUCKET = "public-datasets-lila"
BIG_BIRD_GCS_PREFIX = "big-bird/"
BIG_BIRD_GCS_OBJECT_BASE = f"https://storage.googleapis.com/{BIG_BIRD_GCS_BUCKET}/"
BIG_BIRD_GCS_API = f"https://storage.googleapis.com/storage/v1/b/{BIG_BIRD_GCS_BUCKET}/o"

# The publisher page currently says 4,284 images, while its versioned metadata
# contains 4,824.  These locks follow the machine-readable v2026.03.21 source.
BIG_BIRD_EXPECTED_COUNTS = {
    "images": 4_824,
    "annotations": 53_199,
    "categories": 103,
    "specific_crops": 47_679,
    "generic_bird_crops": 2_311,
    "empty_annotations": 3_209,
    "specific_source_images": 1_514,
    "specific_source_categories": 101,
    "canonical_birdnet_taxa": 100,
}

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_OFFICIAL_METADATA_HOST = "lilawildlife.blob.core.windows.net"
_GCS_HOST = "storage.googleapis.com"


@dataclass(frozen=True, slots=True)
class BigBirdTaxonomyOverride:
    source_scientific_name: str
    source_common_name: str
    birdnet_id: str
    accepted_scientific_name: str
    accepted_common_name: str
    relationship: str
    authority_record: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


# Big Bird treats the Antarctic Shag as a species. AviList v2025b (and hence
# BirdNET Jul-2026) places it under Imperial Shag. This one reviewed alignment
# is exact and version locked; no fuzzy match is allowed.
BIG_BIRD_BIRDNET_TAXONOMY_OVERRIDES: tuple[BigBirdTaxonomyOverride, ...] = (
    BigBirdTaxonomyOverride(
        source_scientific_name="Leucocarbo bransfieldensis",
        source_common_name="antarctic shag",
        birdnet_id="BN16217",
        accepted_scientific_name="Leucocarbo atriceps",
        accepted_common_name="Imperial Shag",
        relationship="species_lumped_as_subspecies",
        authority_record="AviList v2025b accepted species Leucocarbo atriceps",
    ),
)
_OVERRIDE_BY_SOURCE_SCIENTIFIC = {
    override.source_scientific_name.casefold(): override
    for override in BIG_BIRD_BIRDNET_TAXONOMY_OVERRIDES
}


@dataclass(frozen=True, slots=True)
class GCSObjectRecord:
    """Publisher checksum metadata for one source object."""

    object_name: str
    md5_base64: str
    size: int
    generation: str
    content_type: str

    def __post_init__(self) -> None:
        validate_relative_posix_path(self.object_name)
        if not self.object_name.startswith(BIG_BIRD_GCS_PREFIX):
            raise ValueError(f"Big Bird GCS object has an unexpected prefix: {self.object_name}")
        try:
            decoded = base64.b64decode(self.md5_base64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("GCS md5Hash is not valid base64") from error
        if len(decoded) != 16:
            raise ValueError("GCS md5Hash must encode exactly 16 bytes")
        if not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("GCS object size must be positive")
        if not self.generation.strip() or not self.content_type.strip():
            raise ValueError("GCS generation and content_type must not be empty")

    @property
    def md5_hex(self) -> str:
        return base64.b64decode(self.md5_base64, validate=True).hex()

    @property
    def uri(self) -> str:
        return BIG_BIRD_GCS_OBJECT_BASE + quote(self.object_name, safe="/")

    @classmethod
    def from_api(cls, row: Mapping[str, Any]) -> GCSObjectRecord:
        required = {"name", "md5Hash", "size", "generation", "contentType"}
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"GCS object metadata is missing fields: {sorted(missing)}")
        try:
            size = int(row["size"])
        except (TypeError, ValueError) as error:
            raise RuntimeError("GCS object size is not an integer") from error
        return cls(
            object_name=str(row["name"]),
            md5_base64=str(row["md5Hash"]),
            size=size,
            generation=str(row["generation"]),
            content_type=str(row["contentType"]),
        )


@dataclass(frozen=True, slots=True)
class BigBirdImagePlan:
    image_id: str
    file_name: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class BigBirdCropPlan:
    annotation_id: str
    image_id: str
    category_id: int
    raw_label: str
    source_scientific_name: str
    taxon_id: str
    bbox_xywh: tuple[float, float, float, float]
    bbox_xyxy_original: tuple[float, float, float, float]
    bbox_xyxy_effective: tuple[float, float, float, float]
    bbox_was_clipped: bool


@dataclass(frozen=True, slots=True)
class BigBirdMetadataPlan:
    metadata_path: Path
    metadata_sha256: str
    birdnet_csv_sha256: str
    source_version: str
    images: tuple[BigBirdImagePlan, ...]
    crops: tuple[BigBirdCropPlan, ...]
    taxa: tuple[BirdTaxon, ...]
    total_annotation_count: int
    total_category_count: int
    generic_bird_crop_count: int
    empty_annotation_count: int
    specific_source_category_count: int

    @property
    def image_by_id(self) -> dict[str, BigBirdImagePlan]:
        return {image.image_id: image for image in self.images}


@dataclass(frozen=True, slots=True)
class BigBirdPreparation:
    metadata_sha256: str
    birdnet_csv_sha256: str
    source_image_count: int
    downloaded_source_images: int
    reused_source_images: int
    crop_count: int
    written_crops: int
    reused_crops: int
    source_category_count: int
    canonical_taxon_count: int
    clipped_bbox_count: int
    generic_bird_crops_omitted: int
    manifest_fingerprint: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BigBirdStrictCubAudit:
    """Non-mutating report for a strict target-taxon exclusion set."""

    total_crops: int
    total_taxa: int
    total_source_labels: int
    excluded_crops: int
    excluded_taxa: int
    excluded_source_labels: int
    retained_crops: int
    retained_taxa: int
    retained_source_labels: int
    excluded_counts_by_taxon: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["excluded_counts_by_taxon"] = dict(self.excluded_counts_by_taxon)
        return value


@dataclass(frozen=True, slots=True)
class _BirdNetSpecies:
    birdnet_id: str
    scientific_name: str
    common_name: str


@dataclass(frozen=True, slots=True)
class _ProcessedImage:
    samples: tuple[BirdSample, ...]
    provenance: tuple[dict[str, Any], ...]
    source_image: dict[str, Any]
    reused_source: bool
    written_crops: int
    reused_crops: int


def _safe_identifier(value: Any, *, field: str) -> str:
    if value is None or isinstance(value, bool):
        raise RuntimeError(f"Unsafe Big Bird {field}: {value!r}")
    result = str(value)
    if not _SAFE_IDENTIFIER.fullmatch(result):
        raise RuntimeError(f"Unsafe Big Bird {field}: {result!r}")
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"Big Bird {field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Big Bird {field} must be a positive integer") from error
    if result <= 0 or result != value:
        raise RuntimeError(f"Big Bird {field} must be a positive integer")
    return result


def _category_id(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"Big Bird {field} must be a non-negative integer")
    return value


def _contained_output_path(root: Path, relative_path: str) -> Path:
    """Reject output paths that escape the root or traverse any existing symlink."""

    validate_relative_posix_path(relative_path)
    resolved_root = root.resolve()
    unresolved = resolved_root / relative_path
    current = resolved_root
    for part in Path(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Big Bird output path contains a symlink: {relative_path}")
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"Big Bird output path escapes dataset root: {relative_path}") from error
    return unresolved


def _normalise_scientific(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _load_birdnet_species(
    csv_path: Path,
) -> tuple[dict[str, set[str]], dict[str, _BirdNetSpecies]]:
    by_scientific: dict[str, set[str]] = defaultdict(set)
    rows_by_id: dict[str, _BirdNetSpecies] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "birdnet_id",
            "scientific_name",
            "common_name",
            "taxon_group",
            "record_type",
            "scientific_name_aliases",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"BirdNET CSV is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if row["taxon_group"].strip() != "Aves" or row["record_type"].strip() != "species":
                continue
            birdnet_id = row["birdnet_id"].strip()
            scientific_name = row["scientific_name"].strip()
            common_name = row["common_name"].strip()
            if not birdnet_id or not scientific_name or not common_name:
                raise RuntimeError(f"Incomplete BirdNET species at line {line_number}")
            if birdnet_id in rows_by_id:
                raise RuntimeError(f"BirdNET CSV contains duplicate id: {birdnet_id}")
            rows_by_id[birdnet_id] = _BirdNetSpecies(
                birdnet_id=birdnet_id,
                scientific_name=scientific_name,
                common_name=common_name,
            )
            names = [scientific_name]
            names.extend(re.split(r"[|;]", row.get("scientific_name_aliases", "")))
            for name in names:
                if name.strip():
                    by_scientific[_normalise_scientific(name)].add(birdnet_id)
    if not rows_by_id:
        raise RuntimeError("BirdNET CSV contains no Aves species")
    return by_scientific, rows_by_id


def _resolve_big_bird_taxon(
    source_scientific_name: str,
    source_common_name: str,
    by_scientific: Mapping[str, set[str]],
    rows_by_id: Mapping[str, _BirdNetSpecies],
) -> BirdTaxon:
    override = _OVERRIDE_BY_SOURCE_SCIENTIFIC.get(source_scientific_name.casefold())
    if override is not None:
        if source_common_name.casefold() != override.source_common_name.casefold():
            raise RuntimeError(
                "Big Bird reviewed taxonomy override common name changed for "
                f"{source_scientific_name!r}: {source_common_name!r}"
            )
        row = rows_by_id.get(override.birdnet_id)
        if row is None:
            raise RuntimeError(
                f"Reviewed Big Bird BirdNET id is missing: {override.birdnet_id}"
            )
        if (
            row.scientific_name != override.accepted_scientific_name
            or row.common_name != override.accepted_common_name
        ):
            raise RuntimeError(
                "Reviewed Big Bird taxonomy override changed in BirdNET: "
                f"{override.birdnet_id}"
            )
        match = row
    else:
        candidates = by_scientific.get(_normalise_scientific(source_scientific_name), set())
        if len(candidates) != 1:
            raise RuntimeError(
                "Big Bird scientific name must resolve to exactly one BirdNET species: "
                f"{source_scientific_name!r} resolved to {len(candidates)}"
            )
        match = rows_by_id[next(iter(candidates))]
    return BirdTaxon(
        taxon_id=f"birdnet:{match.birdnet_id}",
        scientific_name=match.scientific_name,
        common_name=match.common_name,
        taxonomy_source=BIRDNET_TAXONOMY_AUTHORITY,
        taxonomy_version=BIRDNET_TAXONOMY_VERSION,
    )


def _load_metadata_payload(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Malformed Big Bird metadata: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Big Bird metadata must be a JSON object")
    required = {"info", "images", "annotations", "categories"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"Big Bird metadata is missing keys: {sorted(missing)}")
    if not isinstance(payload["info"], dict) or any(
        not isinstance(payload[key], list) for key in ("images", "annotations", "categories")
    ):
        raise RuntimeError("Big Bird metadata has an invalid top-level schema")
    return payload


def plan_big_bird(
    metadata_path: Path | str,
    *,
    birdnet_csv_path: Path | str,
    require_official_lock: bool = True,
) -> BigBirdMetadataPlan:
    """Validate all metadata and taxonomy before any source image is requested."""

    metadata = Path(metadata_path).expanduser()
    birdnet_csv = Path(birdnet_csv_path).expanduser()
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing Big Bird metadata: {metadata}")
    if not birdnet_csv.is_file():
        raise FileNotFoundError(f"Missing BirdNET taxonomy CSV: {birdnet_csv}")
    metadata_sha256 = sha256_file(metadata)
    birdnet_csv_sha256 = sha256_file(birdnet_csv)
    if require_official_lock and metadata_sha256 != BIG_BIRD_METADATA_OBSERVED_SHA256:
        raise RuntimeError(
            "Big Bird metadata SHA-256 differs from the observed v2026.03.21 object: "
            f"{metadata_sha256}"
        )
    if (
        require_official_lock
        and birdnet_csv_sha256 != BIG_BIRD_BIRDNET_CSV_OBSERVED_SHA256
    ):
        raise RuntimeError(
            "BirdNET CSV SHA-256 differs from the observed v0.3-Jul2026 taxonomy: "
            f"{birdnet_csv_sha256}"
        )
    payload = _load_metadata_payload(metadata)
    info = payload["info"]
    source_version = str(info.get("version", "")).strip()
    if not source_version:
        raise RuntimeError("Big Bird metadata info.version is missing")
    if require_official_lock and source_version != BIG_BIRD_SOURCE_VERSION:
        raise RuntimeError(f"Unexpected Big Bird metadata version: {source_version}")
    if require_official_lock and info.get("license") != BIG_BIRD_LICENSE_URL:
        raise RuntimeError("Big Bird metadata license URL changed")

    categories_by_id: dict[int, str] = {}
    for row in payload["categories"]:
        if not isinstance(row, dict) or "id" not in row or "name" not in row:
            raise RuntimeError("Malformed Big Bird category row")
        category_id = _category_id(row["id"], field="category id")
        name = str(row["name"]).strip()
        if not name:
            raise RuntimeError(f"Big Bird category {category_id} has an empty name")
        if category_id in categories_by_id:
            raise RuntimeError(f"Duplicate Big Bird category id: {category_id}")
        categories_by_id[category_id] = name
    normalised_category_names = [name.casefold() for name in categories_by_id.values()]
    if len(normalised_category_names) != len(set(normalised_category_names)):
        raise RuntimeError("Big Bird category names must be unique ignoring case")
    generic_ids = {key for key, name in categories_by_id.items() if name.casefold() == "bird"}
    empty_ids = {key for key, name in categories_by_id.items() if name.casefold() == "empty"}
    if len(generic_ids) != 1 or len(empty_ids) != 1:
        raise RuntimeError(
            "Big Bird metadata must contain exactly one 'bird' and one 'empty' category"
        )

    images_by_id: dict[str, BigBirdImagePlan] = {}
    file_names: set[str] = set()
    for row in payload["images"]:
        if not isinstance(row, dict):
            raise RuntimeError("Malformed Big Bird image row")
        image_id = _safe_identifier(row.get("id"), field="image id")
        file_name = validate_relative_posix_path(str(row.get("file_name", "")))
        width = _positive_int(row.get("width"), field="image width")
        height = _positive_int(row.get("height"), field="image height")
        if image_id in images_by_id:
            raise RuntimeError(f"Duplicate Big Bird image id: {image_id}")
        if file_name in file_names:
            raise RuntimeError(f"Duplicate Big Bird image path: {file_name}")
        file_names.add(file_name)
        images_by_id[image_id] = BigBirdImagePlan(image_id, file_name, width, height)

    annotation_ids: set[str] = set()
    category_scientific: dict[int, set[str]] = defaultdict(set)
    specific_rows: list[dict[str, Any]] = []
    generic_bird_crop_count = 0
    empty_annotation_count = 0
    for row in payload["annotations"]:
        if not isinstance(row, dict):
            raise RuntimeError("Malformed Big Bird annotation row")
        annotation_id = _safe_identifier(row.get("id"), field="annotation id")
        if annotation_id in annotation_ids:
            raise RuntimeError(f"Duplicate Big Bird annotation id: {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = _safe_identifier(row.get("image_id"), field="annotation image id")
        image = images_by_id.get(image_id)
        if image is None:
            raise RuntimeError(f"Big Bird annotation references unknown image: {image_id}")
        try:
            category_value = row["category_id"]
        except KeyError as error:
            raise RuntimeError(f"Missing category id in annotation {annotation_id}") from error
        category_id = _category_id(category_value, field="annotation category id")
        if category_id not in categories_by_id:
            raise RuntimeError(f"Big Bird annotation references unknown category: {category_id}")
        if row.get("sequence_level_annotation") not in {False}:
            raise RuntimeError(f"Unexpected sequence-level annotation: {annotation_id}")

        if category_id in empty_ids:
            if "bbox" in row:
                raise RuntimeError(f"Empty annotation unexpectedly has bbox: {annotation_id}")
            empty_annotation_count += 1
            continue
        if "bbox" not in row:
            raise RuntimeError(f"Non-empty Big Bird annotation has no bbox: {annotation_id}")
        bbox_value = row["bbox"]
        if not isinstance(bbox_value, (list, tuple)) or len(bbox_value) != 4:
            raise RuntimeError(f"Malformed bbox in Big Bird annotation: {annotation_id}")
        if any(isinstance(value, bool) for value in bbox_value):
            raise RuntimeError(f"Boolean coordinate in Big Bird annotation: {annotation_id}")
        bbox_xywh = tuple(float(value) for value in bbox_value)
        bbox_original = coco_bbox_to_xyxy(bbox_xywh)
        bbox_effective, bbox_was_clipped = clip_bbox_xyxy_to_image(
            bbox_original,
            (image.width, image.height),
        )
        if category_id in generic_ids:
            generic_bird_crop_count += 1
            continue
        genus = str(row.get("genus", "")).strip()
        species = str(row.get("species", "")).strip()
        if not genus or not species or "unknown" in {genus.casefold(), species.casefold()}:
            raise RuntimeError(f"Specific Big Bird annotation lacks taxonomy: {annotation_id}")
        source_scientific_name = f"{genus[0].upper()}{genus[1:].lower()} {species.lower()}"
        category_scientific[category_id].add(source_scientific_name)
        specific_rows.append(
            {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox_xywh": bbox_xywh,
                "bbox_xyxy_original": bbox_original,
                "bbox_xyxy_effective": bbox_effective,
                "bbox_was_clipped": bbox_was_clipped,
            }
        )

    specific_category_ids = set(categories_by_id) - generic_ids - empty_ids
    if set(category_scientific) != specific_category_ids:
        missing = sorted(specific_category_ids - set(category_scientific))
        extra = sorted(set(category_scientific) - specific_category_ids)
        raise RuntimeError(
            f"Big Bird specific category coverage mismatch; missing={missing}, extra={extra}"
        )
    conflicting = {
        category_id: sorted(names)
        for category_id, names in category_scientific.items()
        if len(names) != 1
    }
    if conflicting:
        raise RuntimeError(f"Big Bird category has conflicting scientific names: {conflicting}")

    by_scientific, birdnet_rows = _load_birdnet_species(birdnet_csv)
    taxon_by_category = {
        category_id: _resolve_big_bird_taxon(
            next(iter(category_scientific[category_id])),
            categories_by_id[category_id],
            by_scientific,
            birdnet_rows,
        )
        for category_id in sorted(specific_category_ids)
    }
    crops = tuple(
        BigBirdCropPlan(
            annotation_id=row["annotation_id"],
            image_id=row["image_id"],
            category_id=row["category_id"],
            raw_label=categories_by_id[row["category_id"]],
            source_scientific_name=next(iter(category_scientific[row["category_id"]])),
            taxon_id=taxon_by_category[row["category_id"]].taxon_id,
            bbox_xywh=row["bbox_xywh"],
            bbox_xyxy_original=row["bbox_xyxy_original"],
            bbox_xyxy_effective=row["bbox_xyxy_effective"],
            bbox_was_clipped=row["bbox_was_clipped"],
        )
        for row in sorted(specific_rows, key=lambda value: value["annotation_id"])
    )
    selected_image_ids = {crop.image_id for crop in crops}
    selected_images = tuple(
        sorted(
            (images_by_id[image_id] for image_id in selected_image_ids),
            key=lambda image: image.image_id,
        )
    )
    unique_taxa = {taxon.taxon_id: taxon for taxon in taxon_by_category.values()}
    taxa = tuple(sorted(unique_taxa.values(), key=lambda taxon: taxon.taxon_id))

    observed_counts = {
        "images": len(images_by_id),
        "annotations": len(annotation_ids),
        "categories": len(categories_by_id),
        "specific_crops": len(crops),
        "generic_bird_crops": generic_bird_crop_count,
        "empty_annotations": empty_annotation_count,
        "specific_source_images": len(selected_images),
        "specific_source_categories": len(specific_category_ids),
        "canonical_birdnet_taxa": len(taxa),
    }
    if require_official_lock and observed_counts != BIG_BIRD_EXPECTED_COUNTS:
        raise RuntimeError(
            f"Unexpected Big Bird v2026.03.21 counts: {observed_counts}; "
            f"expected {BIG_BIRD_EXPECTED_COUNTS}"
        )
    del payload
    return BigBirdMetadataPlan(
        metadata_path=metadata.resolve(),
        metadata_sha256=metadata_sha256,
        birdnet_csv_sha256=birdnet_csv_sha256,
        source_version=source_version,
        images=selected_images,
        crops=crops,
        taxa=taxa,
        total_annotation_count=len(annotation_ids),
        total_category_count=len(categories_by_id),
        generic_bird_crop_count=generic_bird_crop_count,
        empty_annotation_count=empty_annotation_count,
        specific_source_category_count=len(specific_category_ids),
    )


def audit_big_bird_strict_cub(
    rows: Sequence[BigBirdCropPlan | BirdSample],
    excluded_birdnet_ids: Iterable[str],
) -> BigBirdStrictCubAudit:
    """Compute target-exclusion counts without altering the canonical manifest."""

    if not rows:
        raise ValueError("Big Bird audit rows must not be empty")
    excluded_taxon_ids = {
        value if value.startswith("birdnet:") else f"birdnet:{value}"
        for raw in excluded_birdnet_ids
        if (value := str(raw).strip())
    }
    all_taxa = {row.taxon_id for row in rows}
    all_labels = {row.raw_label for row in rows}
    excluded_rows = [row for row in rows if row.taxon_id in excluded_taxon_ids]
    retained_rows = [row for row in rows if row.taxon_id not in excluded_taxon_ids]
    excluded_counts: dict[str, int] = defaultdict(int)
    for row in excluded_rows:
        excluded_counts[row.taxon_id] += 1
    return BigBirdStrictCubAudit(
        total_crops=len(rows),
        total_taxa=len(all_taxa),
        total_source_labels=len(all_labels),
        excluded_crops=len(excluded_rows),
        excluded_taxa=len({row.taxon_id for row in excluded_rows}),
        excluded_source_labels=len({row.raw_label for row in excluded_rows}),
        retained_crops=len(retained_rows),
        retained_taxa=len({row.taxon_id for row in retained_rows}),
        retained_source_labels=len({row.raw_label for row in retained_rows}),
        excluded_counts_by_taxon=tuple(sorted(excluded_counts.items())),
    )


def _request_json(url: str, *, attempts: int = 4, timeout: int = 60) -> dict[str, Any]:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != _GCS_HOST:
        raise ValueError(f"Refusing non-GCS API URL: {url}")
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "ttvr-birdmix/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise RuntimeError("GCS API response is not a JSON object")
            return payload
        except urllib.error.HTTPError as error:
            retryable = error.code in {408, 425, 429} or error.code >= 500
            if not retryable or attempt + 1 == attempts:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _list_gcs_objects(required_object_names: set[str]) -> dict[str, GCSObjectRecord]:
    """List publisher metadata once and retain only objects needed by the plan."""

    if not required_object_names:
        raise ValueError("required_object_names must not be empty")
    records: dict[str, GCSObjectRecord] = {}
    page_token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        query: dict[str, str | int] = {
            "prefix": BIG_BIRD_GCS_PREFIX,
            "maxResults": 1000,
            "fields": "nextPageToken,items(name,md5Hash,size,generation,contentType)",
        }
        if page_token is not None:
            query["pageToken"] = page_token
        payload = _request_json(f"{BIG_BIRD_GCS_API}?{urlencode(query)}")
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise RuntimeError("GCS object list items is not a list")
        for row in items:
            if not isinstance(row, dict):
                raise RuntimeError("Malformed GCS object-list row")
            name = str(row.get("name", ""))
            if name not in required_object_names:
                continue
            if name in records:
                raise RuntimeError(f"Duplicate GCS object metadata: {name}")
            records[name] = GCSObjectRecord.from_api(row)
        token_value = payload.get("nextPageToken")
        if not token_value:
            break
        page_token = str(token_value)
        if page_token in seen_tokens:
            raise RuntimeError("GCS object listing repeated a page token")
        seen_tokens.add(page_token)
    missing = sorted(required_object_names - set(records))
    if missing:
        raise RuntimeError(f"GCS listing is missing {len(missing)} required objects: {missing[:5]}")
    return records


def _stream_download_to_path(
    url: str,
    destination: Path,
    *,
    allowed_host: str,
    attempts: int = 4,
    timeout: int = 120,
) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != allowed_host:
        raise ValueError(f"Refusing download from unexpected URL: {url}")
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "ttvr-birdmix/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with destination.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            return
        except urllib.error.HTTPError as error:
            retryable = error.code in {408, 425, 429} or error.code >= 500
            if not retryable or attempt + 1 == attempts:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
        time.sleep(2**attempt)


def _atomic_publish_no_replace(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, destination)
    except FileExistsError:
        return


def _ensure_official_metadata(root: Path) -> Path:
    destination = root / "sources" / BIG_BIRD_METADATA_FILENAME
    if destination.is_file():
        digest = sha256_file(destination)
        if digest != BIG_BIRD_METADATA_OBSERVED_SHA256:
            raise RuntimeError(f"Existing Big Bird metadata SHA-256 mismatch: {digest}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.part-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _stream_download_to_path(
            BIG_BIRD_METADATA_URL,
            temporary,
            allowed_host=_OFFICIAL_METADATA_HOST,
        )
        digest = sha256_file(temporary)
        if digest != BIG_BIRD_METADATA_OBSERVED_SHA256:
            raise RuntimeError(f"Downloaded Big Bird metadata SHA-256 mismatch: {digest}")
        _atomic_publish_no_replace(temporary, destination)
        final_digest = sha256_file(destination)
        if final_digest != BIG_BIRD_METADATA_OBSERVED_SHA256:
            raise RuntimeError(f"Published Big Bird metadata SHA-256 mismatch: {final_digest}")
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def download_big_bird_metadata(
    root: Path | str = Path("data/big_bird"),
) -> Path:
    """Stream and checksum-lock only the official metadata object (no images)."""

    return _ensure_official_metadata(Path(root).expanduser().resolve())


def _ensure_source_image(
    record: GCSObjectRecord,
    destination: Path,
) -> bool:
    """Materialize and MD5-verify one selected source image; return reused status."""

    if destination.is_file():
        actual = md5_base64_file(destination)
        if actual != record.md5_base64:
            raise RuntimeError(
                f"Existing Big Bird source MD5 mismatch for {record.object_name}: {actual}"
            )
        if destination.stat().st_size != record.size:
            raise RuntimeError(f"Existing Big Bird source size mismatch: {record.object_name}")
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.part-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _stream_download_to_path(record.uri, temporary, allowed_host=_GCS_HOST)
        actual = md5_base64_file(temporary)
        if actual != record.md5_base64:
            raise RuntimeError(
                f"Downloaded Big Bird source MD5 mismatch for {record.object_name}: "
                f"expected {record.md5_base64}, found {actual}"
            )
        _atomic_publish_no_replace(temporary, destination)
        # A concurrent winner must represent the same publisher object.
        final_actual = md5_base64_file(destination)
        if final_actual != record.md5_base64:
            raise RuntimeError(
                f"Concurrent Big Bird source MD5 mismatch for {record.object_name}"
            )
        if destination.stat().st_size != record.size:
            raise RuntimeError(f"Downloaded Big Bird source size mismatch: {record.object_name}")
    finally:
        temporary.unlink(missing_ok=True)
    return False


def _reuse_or_write_crop(image: Image.Image, destination: Path) -> tuple[str, str, bool]:
    if destination.is_file():
        with Image.open(destination) as existing_source:
            existing_source.load()
            existing = existing_source.convert("RGB")
        expected = image.convert("RGB")
        if existing.size != expected.size or existing.tobytes() != expected.tobytes():
            raise RuntimeError(f"Existing Big Bird crop does not match recipe: {destination}")
        return sha256_file(destination), perceptual_dhash(existing), True
    try:
        digest, phash = save_crop_png_atomic(image, destination)
        return digest, phash, False
    except FileExistsError:
        # Validate a concurrent winner through the same exact pixel path.
        return _reuse_or_write_crop(image, destination)


def _process_source_image(
    root: Path,
    image_plan: BigBirdImagePlan,
    crop_plans: Sequence[BigBirdCropPlan],
    record: GCSObjectRecord,
    *,
    context_scale: float,
    keep_source_images: bool,
) -> _ProcessedImage:
    source_path = _contained_output_path(
        root,
        f"source_images/{image_plan.file_name}",
    )
    if source_path.is_symlink():
        raise ValueError(f"Big Bird source path must not be a symlink: {source_path}")
    reused_source = _ensure_source_image(record, source_path)
    source_sha256 = sha256_file(source_path)
    samples: list[BirdSample] = []
    provenance: list[dict[str, Any]] = []
    written_crops = 0
    reused_crops = 0
    completed = False
    try:
        with Image.open(source_path) as raw:
            raw.load()
            if raw.size != (image_plan.width, image_plan.height):
                raise RuntimeError(
                    f"Big Bird source dimensions changed for {image_plan.image_id}: "
                    f"metadata={(image_plan.width, image_plan.height)}, image={raw.size}"
                )
            source_image = raw.convert("RGB")
        for crop_plan in sorted(crop_plans, key=lambda crop: crop.annotation_id):
            crop, geometry = square_context_crop(
                source_image,
                crop_plan.bbox_xyxy_effective,
                context_scale=context_scale,
            )
            crop_name = validate_relative_posix_path(f"{crop_plan.annotation_id}.png")
            relative_path = f"crops/{crop_name}"
            destination = _contained_output_path(root, relative_path)
            if destination.is_symlink():
                raise ValueError(f"Big Bird crop path must not be a symlink: {destination}")
            crop_sha256, crop_phash, reused = _reuse_or_write_crop(crop, destination)
            reused_crops += int(reused)
            written_crops += int(not reused)
            samples.append(
                BirdSample(
                    dataset_id=BIG_BIRD_DATASET_ID,
                    source_sample_id=crop_plan.annotation_id,
                    source_split="train",
                    relative_path=relative_path,
                    image_uri=record.uri,
                    group_id=f"big-bird-source-image:{image_plan.image_id}",
                    raw_label=crop_plan.raw_label,
                    taxon_id=crop_plan.taxon_id,
                    sha256=crop_sha256,
                    phash=crop_phash,
                    license=BIG_BIRD_LICENSE,
                    author=BIG_BIRD_AUTHOR,
                    source=BIG_BIRD_SOURCE_URL,
                )
            )
            provenance.append(
                {
                    "annotation_id": crop_plan.annotation_id,
                    "bbox_was_clipped": crop_plan.bbox_was_clipped,
                    "bbox_xywh_official": list(crop_plan.bbox_xywh),
                    "bbox_xyxy_effective": list(crop_plan.bbox_xyxy_effective),
                    "bbox_xyxy_official": list(crop_plan.bbox_xyxy_original),
                    "category_id": crop_plan.category_id,
                    "crop_geometry": geometry.to_dict(),
                    "crop_phash": crop_phash,
                    "crop_relative_path": relative_path,
                    "crop_sha256": crop_sha256,
                    "dataset_id": BIG_BIRD_DATASET_ID,
                    "gcs_generation": record.generation,
                    "gcs_md5_base64": record.md5_base64,
                    "gcs_md5_hex": record.md5_hex,
                    "gcs_object_name": record.object_name,
                    "official_uri": record.uri,
                    "raw_label": crop_plan.raw_label,
                    "source_image_id": image_plan.image_id,
                    "source_image_sha256": source_sha256,
                    "source_scientific_name": crop_plan.source_scientific_name,
                    "taxon_id": crop_plan.taxon_id,
                }
            )
        completed = True
    finally:
        if completed and not keep_source_images:
            source_path.unlink(missing_ok=True)
    source_image_row = {
        "content_type": record.content_type,
        "file_name": image_plan.file_name,
        "gcs_generation": record.generation,
        "gcs_md5_base64": record.md5_base64,
        "gcs_md5_hex": record.md5_hex,
        "gcs_object_name": record.object_name,
        "group_id": f"big-bird-source-image:{image_plan.image_id}",
        "height": image_plan.height,
        "image_id": image_plan.image_id,
        "official_uri": record.uri,
        "size_bytes": record.size,
        "source_image_sha256": source_sha256,
        "specific_crop_count": len(crop_plans),
        "width": image_plan.width,
    }
    return _ProcessedImage(
        samples=tuple(samples),
        provenance=tuple(provenance),
        source_image=source_image_row,
        reused_source=reused_source,
        written_crops=written_crops,
        reused_crops=reused_crops,
    )


def _write_manifest_bundle(
    root: Path,
    *,
    taxa: Sequence[BirdTaxon],
    samples: Sequence[BirdSample],
    provenance: Sequence[dict[str, Any]],
    source_images: Sequence[dict[str, Any]],
    source: dict[str, Any],
) -> None:
    destination = root / "manifests"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace manifest directory: {destination}")
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".manifests.stage-", dir=root))
    lock = root / ".manifests.publish.lock"
    lock_descriptor: int | None = None
    try:
        write_jsonl(stage / "taxa.jsonl", (asdict(taxon) for taxon in taxa))
        write_jsonl(stage / "samples.jsonl", (asdict(sample) for sample in samples))
        write_jsonl(stage / "crop_provenance.jsonl", provenance)
        write_jsonl(stage / "source_images.jsonl", source_images)
        write_json(stage / "source.json", source)
        try:
            lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise FileExistsError(
                f"Another manifest publisher holds the no-replace lock: {lock}"
            ) from error
        if destination.exists():
            raise FileExistsError(f"Refusing to replace manifest directory: {destination}")
        # The O_EXCL lock serializes every cooperating preparer.  With no
        # destination present, rename publishes the complete directory at once.
        os.rename(stage, destination)
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
            lock.unlink(missing_ok=True)
        if stage.exists():
            shutil.rmtree(stage)


def prepare_big_bird(
    root: Path | str = Path("data/big_bird"),
    *,
    birdnet_csv_path: Path | str,
    metadata_path: Path | str | None = None,
    workers: int = 8,
    context_scale: float = DEFAULT_CONTEXT_SCALE,
    keep_source_images: bool = False,
    require_official_lock: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> BigBirdPreparation:
    """Download selected source images, crop every specific box, and lock manifests."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be positive")
    if (
        not isinstance(context_scale, (int, float))
        or not math.isfinite(context_scale)
        or context_scale < 1.0
    ):
        raise ValueError("context_scale must be at least 1.0")
    root_path = Path(root).expanduser().resolve()
    manifests = root_path / "manifests"
    if manifests.exists():
        raise FileExistsError(f"Refusing to replace manifest directory: {manifests}")
    if metadata_path is None:
        if not require_official_lock:
            raise ValueError("non-official preparation requires an explicit metadata_path")
        metadata = download_big_bird_metadata(root_path)
    else:
        metadata = Path(metadata_path).expanduser()
    plan = plan_big_bird(
        metadata,
        birdnet_csv_path=birdnet_csv_path,
        require_official_lock=require_official_lock,
    )

    required_objects = {
        f"{BIG_BIRD_GCS_PREFIX}{image.file_name}" for image in plan.images
    }
    object_records = _list_gcs_objects(required_objects)
    crops_by_image: dict[str, list[BigBirdCropPlan]] = defaultdict(list)
    for crop in plan.crops:
        crops_by_image[crop.image_id].append(crop)

    processed: list[_ProcessedImage] = []
    total = len(plan.images)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_source_image,
                root_path,
                image,
                crops_by_image[image.image_id],
                object_records[f"{BIG_BIRD_GCS_PREFIX}{image.file_name}"],
                context_scale=context_scale,
                keep_source_images=keep_source_images,
            ): image.image_id
            for image in plan.images
        }
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                processed.append(result)
                if progress is not None:
                    progress(completed, total, futures[future])
        except Exception:
            for future in futures:
                future.cancel()
            raise

    samples = tuple(
        sorted(
            (sample for result in processed for sample in result.samples),
            key=lambda sample: sample.source_sample_id,
        )
    )
    provenance = tuple(
        sorted(
            (row for result in processed for row in result.provenance),
            key=lambda row: row["annotation_id"],
        )
    )
    source_images = tuple(
        sorted((result.source_image for result in processed), key=lambda row: row["image_id"])
    )
    if len(samples) != len(plan.crops) or len(provenance) != len(plan.crops):
        raise RuntimeError("Big Bird preparation lost planned crops")
    validation = validate_manifest(
        root_path,
        samples,
        plan.taxa,
        dataset_id=BIG_BIRD_DATASET_ID,
        verify_images=True,
    )
    report = BigBirdPreparation(
        metadata_sha256=plan.metadata_sha256,
        birdnet_csv_sha256=plan.birdnet_csv_sha256,
        source_image_count=len(source_images),
        downloaded_source_images=sum(not result.reused_source for result in processed),
        reused_source_images=sum(result.reused_source for result in processed),
        crop_count=len(samples),
        written_crops=sum(result.written_crops for result in processed),
        reused_crops=sum(result.reused_crops for result in processed),
        source_category_count=plan.specific_source_category_count,
        canonical_taxon_count=len(plan.taxa),
        clipped_bbox_count=sum(crop.bbox_was_clipped for crop in plan.crops),
        generic_bird_crops_omitted=plan.generic_bird_crop_count,
        manifest_fingerprint=validation.fingerprint,
    )
    source = {
        "birdnet_csv_sha256": plan.birdnet_csv_sha256,
        "birdnet_csv_url": BIRDNET_CSV_URL,
        "citation": BIG_BIRD_CITATION,
        "crop_recipe": {
            "context_scale": float(context_scale),
            "format": "lossless PNG",
            "id": SQUARE_CONTEXT_RECIPE_ID,
            "out_of_bounds_policy": (
                "preserve official bbox; explicitly clip to image intersection; record both; "
                "square crop uses zero padding beyond image"
            ),
        },
        "dataset_id": BIG_BIRD_DATASET_ID,
        "distribution_policy": (
            "Derived crops and downloaded source images are private training artifacts; "
            "publish only manifests and attribution/provenance."
        ),
        "gcs": {
            "bucket": BIG_BIRD_GCS_BUCKET,
            "checksum": "per-object GCS md5Hash; publisher provides no bundle checksum",
            "prefix": BIG_BIRD_GCS_PREFIX,
        },
        "generic_category_policy": "omit category name exactly equal to 'bird'",
        "license": BIG_BIRD_LICENSE,
        "metadata_observed_sha256": plan.metadata_sha256,
        "metadata_publisher_checksum": None,
        "metadata_url": BIG_BIRD_METADATA_URL,
        "page_count_discrepancy": (
            "Publisher page says 4,284 images; machine-readable v2026.03.21 metadata has 4,824."
        ),
        "report": report.to_dict(),
        "source_url": BIG_BIRD_SOURCE_URL,
        "source_version": plan.source_version,
        "taxonomy_alignment": {
            "authority": BIRDNET_TAXONOMY_AUTHORITY,
            "birdnet_version": BIRDNET_TAXONOMY_VERSION,
            "method": (
                "unique exact scientific-name or scientific-alias match; reviewed override only; "
                "ambiguity and absence fail closed"
            ),
            "overrides": [
                override.to_dict() for override in BIG_BIRD_BIRDNET_TAXONOMY_OVERRIDES
            ],
        },
    }
    _write_manifest_bundle(
        root_path,
        taxa=plan.taxa,
        samples=samples,
        provenance=provenance,
        source_images=source_images,
        source=source,
    )
    return report
