"""Exact, version-locked source-label alignments to BirdNET/AviList taxa."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .bird_manifest import BirdTaxon, sha256_file
from .birdnet import BIRDNET_TAXONOMY_AUTHORITY, BIRDNET_TAXONOMY_VERSION


@dataclass(frozen=True, slots=True)
class BirdNetTaxonLock:
    source_label: str
    birdnet_id: str
    scientific_name: str
    common_name: str

    @property
    def taxon_id(self) -> str:
        return f"birdnet:{self.birdnet_id}"


def resolve_locked_birdnet_taxa(
    csv_path: Path | str,
    locks: Iterable[BirdNetTaxonLock],
    *,
    expected_sha256: str,
    require_official_lock: bool = True,
) -> tuple[dict[str, BirdTaxon], str]:
    """Resolve reviewed labels and reject any taxonomy drift or ambiguity."""

    path = Path(csv_path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Missing regular BirdNET taxonomy CSV: {path}")
    digest = sha256_file(path)
    if require_official_lock and digest != expected_sha256:
        raise RuntimeError(
            "BirdNET CSV SHA-256 differs from the reviewed taxonomy: "
            f"expected {expected_sha256}, found {digest}"
        )
    reviewed = tuple(locks)
    if not reviewed:
        raise ValueError("at least one BirdNET taxon lock is required")
    labels = [lock.source_label for lock in reviewed]
    ids = [lock.birdnet_id for lock in reviewed]
    if len(labels) != len(set(labels)) or len(ids) != len(set(ids)):
        raise ValueError("BirdNET taxon lock source labels and ids must be unique")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
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
        rows: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            birdnet_id = row["birdnet_id"].strip()
            if birdnet_id not in ids:
                continue
            if birdnet_id in rows:
                raise RuntimeError(
                    f"BirdNET CSV duplicates reviewed id {birdnet_id} at line {line_number}"
                )
            rows[birdnet_id] = row

    result: dict[str, BirdTaxon] = {}
    for lock in reviewed:
        row = rows.get(lock.birdnet_id)
        if row is None:
            raise RuntimeError(f"Reviewed BirdNET id is missing: {lock.birdnet_id}")
        observed = (
            row["scientific_name"].strip(),
            row["common_name"].strip(),
            row["taxon_group"].strip(),
            row["record_type"].strip(),
        )
        expected = (lock.scientific_name, lock.common_name, "Aves", "species")
        if observed != expected:
            raise RuntimeError(
                f"Reviewed BirdNET taxon changed for {lock.birdnet_id}: "
                f"expected {expected}, found {observed}"
            )
        result[lock.source_label] = BirdTaxon(
            taxon_id=lock.taxon_id,
            scientific_name=lock.scientific_name,
            common_name=lock.common_name,
            taxonomy_source=BIRDNET_TAXONOMY_AUTHORITY,
            taxonomy_version=BIRDNET_TAXONOMY_VERSION,
        )
    return result, digest
