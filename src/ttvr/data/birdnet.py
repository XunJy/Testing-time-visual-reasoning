"""BirdNET+ Taxonomy image source with per-image licence filtering."""

from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from .bird_manifest import (
    BirdSample,
    BirdTaxon,
    perceptual_dhash,
    sha256_file,
    write_json,
    write_jsonl,
)

BIRDNET_TAXONOMY_VERSION = "v0.3-Jul2026"
BIRDNET_DATASET_ID = "birdnet-taxonomy-v0.3-jul2026"
BIRDNET_CSV_URL = "https://birdnet.cornell.edu/taxonomy/api/download/csv"
BIRDNET_TAXONOMY_URL = "https://birdnet.cornell.edu/taxonomy/download"
BIRDNET_TAXONOMY_AUTHORITY = "AviList via BirdNET+ Taxonomy"
_BIRDNET_DOWNLOAD_HOST = "birdnet.cornell.edu"


@dataclass(frozen=True, slots=True)
class BirdNetPreparation:
    csv_sha256: str
    requested: int
    downloaded: int
    reused: int
    failed: int
    excluded_no_image: int
    excluded_license: int
    taxonomy_count: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def license_permits_noncommercial_training(value: str) -> bool:
    """Conservatively accept CC/PD images without a no-derivatives clause."""

    normalised = " ".join(value.strip().casefold().replace("_", "-").split())
    if not normalised or "©" in value or "no derivatives" in normalised:
        return False
    compact = normalised.replace(" ", "-")
    if "-nd" in compact or compact.endswith("nd"):
        return False
    public_domain = re.fullmatch(
        r"(?:cc0|pd|public-domain|gfdl)(?:-[0-9]+(?:\.[0-9]+)?)?",
        compact,
    )
    creative_commons = re.fullmatch(
        r"cc-by(?:-nc)?(?:-sa)?(?:-[0-9]+(?:\.[0-9]+)?)?",
        compact,
    )
    return public_domain is not None or creative_commons is not None


def _download(
    url: str,
    destination: Path,
    *,
    timeout: int = 60,
    attempts: int = 3,
) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != _BIRDNET_DOWNLOAD_HOST:
        raise ValueError(f"Refusing non-BirdNET download URL: {url}")
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.part-",
        dir=destination.parent,
        delete=False,
    ) as temporary_handle:
        temporary = Path(temporary_handle.name)
    try:
        for attempt in range(attempts):
            request = urllib.request.Request(
                url, headers={"User-Agent": "ttvr-birdmix/1.0"}
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
                break
            except urllib.error.HTTPError as error:
                retryable = error.code in {408, 425, 429} or error.code >= 500
                if not retryable or attempt + 1 == attempts:
                    raise
            except (TimeoutError, urllib.error.URLError):
                if attempt + 1 == attempts:
                    raise
            time.sleep(2**attempt)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_image_metadata(path: Path) -> tuple[str, str]:
    with Image.open(path) as image:
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise RuntimeError(f"Invalid image dimensions: {path}")
        phash = perceptual_dhash(image)
    return sha256_file(path), phash


def _image_provenance(row: dict[str, str]) -> dict[str, str]:
    return {
        "birdnet_id": row["birdnet_id"],
        "image_author": row["image_author"],
        "image_license": row["image_license"],
        "image_source": row["image_source"],
        "image_url": row["image_url"],
    }


def _provenance_matches(path: Path, expected: dict[str, str]) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value == expected


def _download_one(row: dict[str, str], image_root: Path) -> tuple[BirdSample, bool]:
    birdnet_id = row["birdnet_id"]
    destination = image_root / f"{birdnet_id}.webp"
    provenance_path = image_root / ".provenance" / f"{birdnet_id}.json"
    expected_provenance = _image_provenance(row)
    reused = destination.is_file() and _provenance_matches(
        provenance_path, expected_provenance
    )
    if not reused:
        destination.unlink(missing_ok=True)
        provenance_path.unlink(missing_ok=True)
        _download(row["image_url"], destination)
    try:
        digest, phash = _validated_image_metadata(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        provenance_path.unlink(missing_ok=True)
        if reused:
            _download(row["image_url"], destination)
            digest, phash = _validated_image_metadata(destination)
            reused = False
        else:
            raise
    if not reused:
        write_json(provenance_path, expected_provenance)
    return (
        BirdSample(
            dataset_id=BIRDNET_DATASET_ID,
            source_sample_id=birdnet_id,
            source_split="train",
            relative_path=f"images/{destination.name}",
            image_uri=row["image_url"],
            group_id=f"birdnet-species-image:{birdnet_id}",
            raw_label=row["common_name"],
            taxon_id=f"birdnet:{birdnet_id}",
            sha256=digest,
            phash=phash,
            license=row["image_license"],
            author=row["image_author"].strip() or "unknown (source metadata empty)",
            source=row["image_source"],
        ),
        reused,
    )


def prepare_birdnet_training_images(
    root: Path | str,
    *,
    workers: int = 16,
    max_samples: int | None = None,
) -> BirdNetPreparation:
    """Download the licensed global one-image-per-species BirdNET source.

    This is an auxiliary training source, not a standard classification
    benchmark.  The generated manifests lock the exact successful subset and
    retain attribution and licence metadata for every image.
    """

    if workers <= 0 or (max_samples is not None and max_samples <= 0):
        raise ValueError("workers and max_samples must be positive")
    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    manifests = root_path / "manifests"
    if manifests.exists():
        raise FileExistsError(f"Refusing to replace manifest directory: {manifests}")
    csv_path = root_path / f"birdnet-taxonomy-{BIRDNET_TAXONOMY_VERSION}.csv"
    if not csv_path.is_file():
        _download(BIRDNET_CSV_URL, csv_path)
    csv_digest = sha256_file(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "birdnet_id",
            "common_name",
            "image_author",
            "image_license",
            "image_source",
            "image_url",
            "record_type",
            "scientific_name",
            "taxon_group",
        }
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise RuntimeError(f"BirdNET CSV is missing columns: {sorted(missing_columns)}")
        rows = list(reader)
    birds = [
        row
        for row in rows
        if row.get("taxon_group") == "Aves" and row.get("record_type") == "species"
    ]
    bird_ids = [row["birdnet_id"] for row in birds]
    if len(bird_ids) != len(set(bird_ids)):
        raise RuntimeError("BirdNET CSV contains duplicate bird species ids")
    no_image = sum(not row.get("image_url", "").strip() for row in birds)
    with_images = [row for row in birds if row.get("image_url", "").strip()]
    allowed = [
        row
        for row in with_images
        if license_permits_noncommercial_training(row.get("image_license", ""))
    ]
    excluded_license = len(with_images) - len(allowed)
    allowed.sort(key=lambda row: row["birdnet_id"])
    if max_samples is not None:
        allowed = allowed[:max_samples]

    image_root = root_path / "images"
    samples: list[BirdSample] = []
    failures: list[dict[str, str]] = []
    reused_count = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, row, image_root): row for row in allowed}
        for future in as_completed(futures):
            row = futures[future]
            try:
                sample, reused = future.result()
                samples.append(sample)
                reused_count += int(reused)
            except Exception as error:
                failures.append(
                    {
                        "birdnet_id": row["birdnet_id"],
                        "image_url": row["image_url"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    samples.sort(key=lambda sample: sample.source_sample_id)
    if not samples:
        raise RuntimeError("No BirdNET images were downloaded successfully")
    successful_ids = {sample.source_sample_id for sample in samples}
    row_by_id = {row["birdnet_id"]: row for row in allowed}
    taxa = [
        BirdTaxon(
            taxon_id=f"birdnet:{birdnet_id}",
            scientific_name=row_by_id[birdnet_id]["scientific_name"],
            common_name=row_by_id[birdnet_id]["common_name"],
            taxonomy_source=BIRDNET_TAXONOMY_AUTHORITY,
            taxonomy_version=BIRDNET_TAXONOMY_VERSION,
        )
        for birdnet_id in sorted(successful_ids)
    ]
    write_jsonl(manifests / "taxa.jsonl", (asdict(taxon) for taxon in taxa))
    write_jsonl(manifests / "samples.jsonl", (asdict(sample) for sample in samples))
    if failures:
        write_jsonl(manifests / "download_failures.jsonl", failures)
    report = BirdNetPreparation(
        csv_sha256=csv_digest,
        requested=len(allowed),
        downloaded=len(samples) - reused_count,
        reused=reused_count,
        failed=len(failures),
        excluded_no_image=no_image,
        excluded_license=excluded_license,
        taxonomy_count=len(birds),
    )
    source = {
        "dataset_id": BIRDNET_DATASET_ID,
        "dataset_role": "auxiliary_training_source_not_benchmark",
        "source_url": BIRDNET_TAXONOMY_URL,
        "csv_url": BIRDNET_CSV_URL,
        "taxonomy_version": BIRDNET_TAXONOMY_VERSION,
        "usage_scope": "noncommercial-research",
        "license_policy": "CC, GFDL, or public-domain; no ND; no Macaulay copyright",
        "report": report.to_dict(),
    }
    write_json(manifests / "source.json", source)
    return report


def load_birdnet_manifests(
    root: Path | str,
) -> tuple[tuple[BirdSample, ...], tuple[BirdTaxon, ...]]:
    """Load a completed BirdNET preparation without touching the network."""

    from .bird_manifest import load_samples, load_taxa

    root_path = Path(root).expanduser()
    return (
        load_samples(root_path / "manifests/samples.jsonl"),
        load_taxa(root_path / "manifests/taxa.jsonl"),
    )
