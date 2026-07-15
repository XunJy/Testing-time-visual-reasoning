"""Official iNaturalist 2021 Mini Aves source for BirdMix."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .bird_manifest import (
    BirdSample,
    BirdTaxon,
    perceptual_dhash,
    sha256_file,
    write_json,
    write_jsonl,
)
from .birdnet import (
    BIRDNET_CSV_URL,
    BIRDNET_TAXONOMY_AUTHORITY,
    BIRDNET_TAXONOMY_VERSION,
)

INAT2021_DATASET_ID = "inaturalist-2021-mini-aves"
INAT2021_VERSION = "2021-03-02"
INAT2021_ARCHIVE_MD5 = "db6ed8330e634445efc8fec83ae81442"
INAT2021_ANNOTATION_MD5 = "395a35be3651d86dc3b0d365b8ea5f92"
INAT2021_SOURCE_URL = "https://github.com/visipedia/inat_comp/tree/master/2021"
INAT2021_ARCHIVE_URL = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz"
)
INAT2021_ANNOTATION_URL = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz"
)
INAT2021_BIRDNET_TAXONOMY_ALIGNMENT_PROTOCOL = "inat2021-mini-avilist-v2025b-reviewed-overrides-v1"
INAT2021_BIRDNET_TAXONOMY_AUTHORITY_URL = "https://www.avilist.org/checklist/v2025b/"


@dataclass(frozen=True, slots=True)
class INatBirdNetTaxonomyOverride:
    """One reviewed alignment from the 2021 source taxonomy to AviList v2025b."""

    source_scientific_name: str
    source_common_name: str
    birdnet_id: str
    accepted_scientific_name: str
    accepted_common_name: str
    relationship: str
    authority_record: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


# iNaturalist 2021 predates AviList and six source species do not occur as
# aliases in the Jul-2026 BirdNET CSV.  These exact, version-locked records are
# deliberately reviewed rather than fuzzy matched.  In particular, historical
# Northwestern Crow is now within American Crow, a CUB target species; leaving
# it as ``inat2021:3750`` would bypass strict ``birdnet:*`` target exclusion.
INAT2021_BIRDNET_TAXONOMY_OVERRIDES: tuple[INatBirdNetTaxonomyOverride, ...] = (
    INatBirdNetTaxonomyOverride(
        source_scientific_name="Tetrao tetrix",
        source_common_name="Eurasian Black Grouse",
        birdnet_id="BN08208",
        accepted_scientific_name="Lyrurus tetrix",
        accepted_common_name="Black Grouse",
        relationship="genus_change",
        authority_record="AviList v2025b proposal 2025-0131",
    ),
    INatBirdNetTaxonomyOverride(
        source_scientific_name="Corvus caurinus",
        source_common_name="Northwestern Crow",
        birdnet_id="BN03722",
        accepted_scientific_name="Corvus brachyrhynchos",
        accepted_common_name="American Crow",
        relationship="species_lumped_as_subspecies",
        authority_record="AviList v2025b proposal 2025-1081",
    ),
    INatBirdNetTaxonomyOverride(
        source_scientific_name="Acanthis hornemanni",
        source_common_name="Hoary Redpoll",
        birdnet_id="BN00011",
        accepted_scientific_name="Acanthis flammea",
        accepted_common_name="Redpoll",
        relationship="species_lumped_as_subspecies",
        authority_record="AviList v2025b proposal 2025-1108",
    ),
    INatBirdNetTaxonomyOverride(
        source_scientific_name="Parus minor",
        source_common_name="Japanese Tit",
        birdnet_id="BN10706",
        accepted_scientific_name="Parus cinereus",
        accepted_common_name="Cinereous Tit",
        relationship="species_lumped_as_subspecies",
        authority_record="AviList v2025b proposal 2025-1100",
    ),
    INatBirdNetTaxonomyOverride(
        source_scientific_name="Sylvia communis",
        source_common_name="Whitethroat",
        birdnet_id="BN04073",
        accepted_scientific_name="Curruca communis",
        accepted_common_name="Common Whitethroat",
        relationship="genus_change",
        authority_record="AviList v2025b species row 23810 (protonym Sylvia communis)",
    ),
    INatBirdNetTaxonomyOverride(
        source_scientific_name="Empidonax occidentalis",
        source_common_name="Cordilleran Flycatcher",
        birdnet_id="BN05095",
        accepted_scientific_name="Empidonax difficilis",
        accepted_common_name="Western Flycatcher",
        relationship="species_lumped_as_subspecies",
        authority_record="AviList v2025b proposal 2025-0922",
    ),
)

_INAT2021_BIRDNET_OVERRIDE_BY_SOURCE_NAME = {
    override.source_scientific_name: override for override in INAT2021_BIRDNET_TAXONOMY_OVERRIDES
}
if len(_INAT2021_BIRDNET_OVERRIDE_BY_SOURCE_NAME) != len(
    INAT2021_BIRDNET_TAXONOMY_OVERRIDES
):  # pragma: no cover - module-level invariant.
    raise RuntimeError("Duplicate iNaturalist taxonomy override source name")


@dataclass(frozen=True, slots=True)
class INatPreparation:
    archive_md5: str
    annotation_md5: str
    birdnet_csv_sha256: str
    category_count: int
    canonical_taxon_count: int
    sample_count: int
    training_count: int
    validation_count: int
    birdnet_mapped_categories: int
    source_taxonomy_categories: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - comparison with the publisher's checksum.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _load_annotation_payload(annotation_archive: Path) -> dict[str, Any]:
    """Load the sole official JSON member from the checksum-verified archive."""

    with tarfile.open(annotation_archive, mode="r:gz") as archive:
        try:
            member = archive.getmember("train_mini.json")
        except KeyError as error:
            raise RuntimeError("iNat annotation archive is missing train_mini.json") from error
        if not member.isfile():
            raise RuntimeError("iNat train_mini.json archive member is not a regular file")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError("Could not read train_mini.json from annotation archive")
        with extracted:
            payload = json.load(extracted)
    if not isinstance(payload, dict):
        raise RuntimeError("iNat annotation payload must be a JSON object")
    required = {"annotations", "categories", "images", "licenses"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"iNat annotation payload is missing keys: {sorted(missing)}")
    return payload


def _birdnet_indices(
    csv_path: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, dict[str, str]]]:
    by_scientific: dict[str, set[str]] = defaultdict(set)
    by_common: dict[str, set[str]] = defaultdict(set)
    rows_by_id: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "birdnet_id",
            "common_name",
            "record_type",
            "scientific_name",
            "taxon_group",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"BirdNET CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if row.get("taxon_group") != "Aves" or row.get("record_type") != "species":
                continue
            birdnet_id = row["birdnet_id"]
            if birdnet_id in rows_by_id:
                raise RuntimeError(f"BirdNET CSV contains duplicate id: {birdnet_id}")
            rows_by_id[birdnet_id] = row
            scientific_values = [row["scientific_name"]]
            scientific_values.extend(re.split(r"[|;]", row.get("scientific_name_aliases", "")))
            for value in scientific_values:
                if value.strip():
                    by_scientific[value.strip().casefold()].add(birdnet_id)
            common_values = [row["common_name"], row.get("common_name_alt", "")]
            common_values.extend(re.split(r"[|;]", row.get("common_name_aliases", "")))
            for value in common_values:
                if value.strip():
                    by_common[_normalise_name(value)].add(birdnet_id)
    return by_scientific, by_common, rows_by_id


def _resolve_taxon(
    category: dict[str, Any],
    by_scientific: dict[str, set[str]],
    by_common: dict[str, set[str]],
    birdnet_rows: dict[str, dict[str, str]],
) -> BirdTaxon:
    source_scientific_name = str(category["name"])
    source_common_name = str(category["common_name"])
    override = _INAT2021_BIRDNET_OVERRIDE_BY_SOURCE_NAME.get(source_scientific_name)
    if override is not None:
        if source_common_name != override.source_common_name:
            raise RuntimeError(
                "iNaturalist taxonomy override source common name changed for "
                f"{source_scientific_name!r}: {source_common_name!r}"
            )
        row = birdnet_rows.get(override.birdnet_id)
        if row is None:
            raise RuntimeError(
                "Reviewed iNaturalist taxonomy override is missing from BirdNET: "
                f"{override.birdnet_id}"
            )
        if row["scientific_name"] != override.accepted_scientific_name:
            raise RuntimeError(
                "Reviewed iNaturalist taxonomy override changed in BirdNET: "
                f"{override.birdnet_id} expected "
                f"{override.accepted_scientific_name!r}, found "
                f"{row['scientific_name']!r}"
            )
        return BirdTaxon(
            taxon_id=f"birdnet:{override.birdnet_id}",
            scientific_name=row["scientific_name"],
            common_name=row["common_name"],
            taxonomy_source=BIRDNET_TAXONOMY_AUTHORITY,
            taxonomy_version=BIRDNET_TAXONOMY_VERSION,
        )

    candidates = by_scientific.get(source_scientific_name.casefold(), set())
    if len(candidates) != 1:
        candidates = by_common.get(_normalise_name(source_common_name), set())
    if len(candidates) == 1:
        birdnet_id = next(iter(candidates))
        row = birdnet_rows[birdnet_id]
        return BirdTaxon(
            taxon_id=f"birdnet:{birdnet_id}",
            scientific_name=row["scientific_name"],
            common_name=row["common_name"],
            taxonomy_source=BIRDNET_TAXONOMY_AUTHORITY,
            taxonomy_version=BIRDNET_TAXONOMY_VERSION,
        )
    return BirdTaxon(
        taxon_id=f"inat2021:{category['id']}",
        scientific_name=source_scientific_name,
        common_name=source_common_name,
        taxonomy_source="iNaturalist 2021 competition taxonomy",
        taxonomy_version=INAT2021_VERSION,
    )


def _taxonomy_alignment_metadata() -> dict[str, Any]:
    """Return the reviewed alignment protocol for the immutable source manifest."""

    return {
        "protocol": INAT2021_BIRDNET_TAXONOMY_ALIGNMENT_PROTOCOL,
        "authority": "AviList v2025b",
        "authority_url": INAT2021_BIRDNET_TAXONOMY_AUTHORITY_URL,
        "override_count": len(INAT2021_BIRDNET_TAXONOMY_OVERRIDES),
        "overrides": [override.to_dict() for override in INAT2021_BIRDNET_TAXONOMY_OVERRIDES],
    }


def _stable_split(
    image_rows: list[dict[str, Any]],
    annotations_by_image: dict[int, dict[str, Any]],
    *,
    validation_per_class: int,
) -> dict[int, str]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for image in image_rows:
        grouped[int(annotations_by_image[int(image["id"])]["category_id"])].append(image)
    result: dict[int, str] = {}
    salt = "ttvr-inat2021-mini-source-validation-v1"
    for category_id, rows in grouped.items():
        if len(rows) <= validation_per_class:
            raise RuntimeError(f"iNat category {category_id} has too few Mini images")
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(f"{salt}\0{row['file_name']}".encode()).hexdigest(),
        )
        validation_ids = {int(row["id"]) for row in ranked[:validation_per_class]}
        for row in rows:
            result[int(row["id"])] = "validation" if int(row["id"]) in validation_ids else "train"
    return result


def _image_metadata(path: Path) -> tuple[str, str]:
    digest = sha256_file(path)
    with Image.open(path) as image:
        image.load()
        phash = perceptual_dhash(image)
    return digest, phash


def _extract_missing_archive_members(
    archive: Path,
    root: Path,
    expected_names: list[str],
) -> int:
    """Extract only absent members so a verified re-prepare preserves existing files."""

    missing_names = [name for name in expected_names if not (root / name).is_file()]
    if not missing_names:
        return 0
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".ttvr-aves-files-",
        suffix=".txt",
        dir=root,
        delete=False,
    ) as file_list_handle:
        file_list_handle.write("\n".join(missing_names) + "\n")
        file_list = Path(file_list_handle.name)
    try:
        subprocess.run(
            ["tar", "-xzf", str(archive), "-C", str(root), "-T", str(file_list)],
            check=True,
        )
    finally:
        file_list.unlink(missing_ok=True)
    still_missing = [name for name in missing_names if not (root / name).is_file()]
    if still_missing:
        raise FileNotFoundError(f"Extraction missed {len(still_missing)} Aves images")
    return len(missing_names)


def prepare_inat2021_mini_aves(
    root: Path | str,
    *,
    birdnet_csv_path: Path | str,
    validation_per_class: int = 5,
    workers: int = 16,
) -> INatPreparation:
    """Extract only Aves images and lock taxonomy, licence, and split manifests."""

    if validation_per_class <= 0 or workers <= 0:
        raise ValueError("validation_per_class and workers must be positive")
    root_path = Path(root).expanduser()
    archive = root_path / "train_mini.tar.gz"
    annotation_archive = root_path / "train_mini.json.tar.gz"
    manifests = root_path / "manifests"
    if manifests.exists():
        raise FileExistsError(f"Refusing to replace manifest directory: {manifests}")
    for path in (archive, annotation_archive):
        if not path.is_file():
            raise FileNotFoundError(f"Missing iNaturalist source file: {path}")
    archive_md5 = _md5_file(archive)
    annotation_md5 = _md5_file(annotation_archive)
    if archive_md5 != INAT2021_ARCHIVE_MD5:
        raise RuntimeError(f"iNat image archive MD5 mismatch: {archive_md5}")
    if annotation_md5 != INAT2021_ANNOTATION_MD5:
        raise RuntimeError(f"iNat annotation archive MD5 mismatch: {annotation_md5}")

    payload = _load_annotation_payload(annotation_archive)
    categories = [
        category
        for category in payload["categories"]
        if category.get("class") == "Aves" and category.get("supercategory") == "Birds"
    ]
    category_ids_in_source = [int(category["id"]) for category in categories]
    if len(category_ids_in_source) != len(set(category_ids_in_source)):
        raise RuntimeError("iNat Mini contains duplicate Aves category ids")
    category_by_id = {int(category["id"]): category for category in categories}
    category_ids = {int(category["id"]) for category in categories}
    annotations = [
        annotation
        for annotation in payload["annotations"]
        if int(annotation["category_id"]) in category_ids
    ]
    annotations_by_image = {int(row["image_id"]): row for row in annotations}
    if len(annotations_by_image) != len(annotations):
        raise RuntimeError("iNat Mini annotations must contain one label per image")
    images = [image for image in payload["images"] if int(image["id"]) in annotations_by_image]
    license_rows = list(payload["licenses"])
    del payload, annotations
    if len(categories) != 1_486 or len(images) != 74_300:
        raise RuntimeError(
            f"Unexpected iNat Mini Aves counts: {len(categories)} classes, {len(images)} images"
        )

    expected_names = sorted(str(image["file_name"]) for image in images)
    if len(expected_names) != len(set(expected_names)):
        raise RuntimeError("iNaturalist annotation contains duplicate Aves image paths")
    if any(
        Path(name).is_absolute() or ".." in Path(name).parts or "\n" in name or "\r" in name
        for name in expected_names
    ):
        raise RuntimeError("Unsafe image name in iNaturalist annotation")
    _extract_missing_archive_members(archive, root_path, expected_names)

    birdnet_csv = Path(birdnet_csv_path).expanduser()
    if not birdnet_csv.is_file():
        raise FileNotFoundError(f"Missing BirdNET taxonomy CSV: {birdnet_csv}")
    birdnet_csv_sha256 = sha256_file(birdnet_csv)
    by_scientific, by_common, birdnet_rows = _birdnet_indices(birdnet_csv)
    category_taxa = {
        int(category["id"]): _resolve_taxon(
            category,
            by_scientific,
            by_common,
            birdnet_rows,
        )
        for category in categories
    }
    split_by_image = _stable_split(
        images,
        annotations_by_image,
        validation_per_class=validation_per_class,
    )
    licenses = {int(row["id"]): f"{row['name']} ({row['url']})" for row in license_rows}
    paths = [root_path / str(image["file_name"]) for image in images]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        metadata = list(pool.map(_image_metadata, paths))

    samples: list[BirdSample] = []
    for image, (digest, phash) in zip(images, metadata, strict=True):
        image_id = int(image["id"])
        category_id = int(annotations_by_image[image_id]["category_id"])
        taxon = category_taxa[category_id]
        file_name = str(image["file_name"])
        samples.append(
            BirdSample(
                dataset_id=INAT2021_DATASET_ID,
                source_sample_id=str(image_id),
                source_split=split_by_image[image_id],
                relative_path=file_name,
                image_uri=f"inat2021-archive:{file_name}",
                group_id=f"inat2021-image:{image_id}",
                raw_label=str(category_by_id[category_id]["name"]),
                taxon_id=taxon.taxon_id,
                sha256=digest,
                phash=phash,
                license=licenses[int(image["license"])],
                author=str(image.get("rights_holder") or "unknown (source metadata empty)"),
                source="iNaturalist 2021 competition Mini training archive",
            )
        )
    samples.sort(key=lambda sample: int(sample.source_sample_id))
    unique_taxa = {taxon.taxon_id: taxon for taxon in category_taxa.values()}
    taxa = tuple(sorted(unique_taxa.values(), key=lambda taxon: taxon.taxon_id))
    write_jsonl(manifests / "taxa.jsonl", (asdict(taxon) for taxon in taxa))
    write_jsonl(manifests / "samples.jsonl", (asdict(sample) for sample in samples))
    report = INatPreparation(
        archive_md5=archive_md5,
        annotation_md5=annotation_md5,
        birdnet_csv_sha256=birdnet_csv_sha256,
        category_count=len(categories),
        canonical_taxon_count=len(taxa),
        sample_count=len(samples),
        training_count=sum(sample.source_split == "train" for sample in samples),
        validation_count=sum(sample.source_split == "validation" for sample in samples),
        birdnet_mapped_categories=sum(
            taxon.taxon_id.startswith("birdnet:") for taxon in category_taxa.values()
        ),
        source_taxonomy_categories=sum(
            taxon.taxon_id.startswith("inat2021:") for taxon in category_taxa.values()
        ),
    )
    source = {
        "dataset_id": INAT2021_DATASET_ID,
        "source_url": INAT2021_SOURCE_URL,
        "archive_url": INAT2021_ARCHIVE_URL,
        "annotation_url": INAT2021_ANNOTATION_URL,
        "birdnet_csv_url": BIRDNET_CSV_URL,
        "source_version": INAT2021_VERSION,
        "terms": "non-commercial research and education; do not redistribute images",
        "taxonomy_alignment": _taxonomy_alignment_metadata(),
        "derived_split": {
            "method": "SHA256 rank within species",
            "validation_per_class": validation_per_class,
            "salt": "ttvr-inat2021-mini-source-validation-v1",
        },
        "report": report.to_dict(),
    }
    write_json(manifests / "source.json", source)
    return report
