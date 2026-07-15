"""Strict CUB base/novel protocol and BirdNET taxonomy alignment.

The CUB labels are historical English common names.  Most of them can be
aligned to BirdNET with an exact match after harmless punctuation and case
normalisation.  The remaining labels are resolved by a small reviewed table
of current scientific names.  This module deliberately does not use fuzzy
matching: a changed or ambiguous taxonomy must fail loudly instead of
silently leaking a held-out species into adapter training.

Class ids in this module are zero based, matching :class:`CUB200Dataset`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

_CUB_CLASS_COUNT = 200

CUB_BASE_NOVEL_PROTOCOL = "cub-100-100-adjacent-salted-sha256-v1"
CUB_BASE_NOVEL_SPLIT_SALT = "ttvr-cub-adjacent-pairs-2026-07-15"
CUB_BIRDNET_CROSSWALK_PROTOCOL = "cub-birdnet-reviewed-v1"

# These CUB labels do not have a unique normalized English alias in the
# Jul-2026 BirdNET+ Taxonomy.  Values are reviewed current scientific names,
# not fuzzy-search results.  The table covers dropped possessives, spelling
# errors, historical names, and generic CUB labels whose intended North
# American species is well established by the benchmark vocabulary.
CUB_SCIENTIFIC_NAME_OVERRIDES: Mapping[str, str] = MappingProxyType(
    {
        "Brewer Blackbird": "Euphagus cyanocephalus",
        "Cardinal": "Cardinalis cardinalis",
        "Chuck-will Widow": "Antrostomus carolinensis",
        "Brandt Cormorant": "Urile penicillatus",
        "Frigatebird": "Fregata magnificens",
        "Heermann Gull": "Larus heermanni",
        "Herring Gull": "Larus argentatus",
        "Anna Hummingbird": "Calypte anna",
        "Green Violetear": "Colibri thalassinus",
        "Florida Jay": "Aphelocoma coerulescens",
        "White-breasted Kingfisher": "Halcyon smyrnensis",
        "Mockingbird": "Mimus polyglottos",
        "Nighthawk": "Chordeiles minor",
        "Clark Nutcracker": "Nucifraga columbiana",
        "Scott Oriole": "Icterus parisorum",
        "White Pelican": "Pelecanus erythrorhynchos",
        "Whip-poor Will": "Antrostomus vociferus",
        "Geococcyx": "Geococcyx californianus",
        "Baird Sparrow": "Centronyx bairdii",
        "Brewer Sparrow": "Spizella breweri",
        "Harris Sparrow": "Zonotrichia querula",
        "Henslow Sparrow": "Centronyx henslowii",
        "Le-Conte Sparrow": "Ammospiza leconteii",
        "Lincoln Sparrow": "Melospiza lincolnii",
        "Nelson-Sharp-tailed Sparrow": "Ammospiza nelsoni",
        "Tree Sparrow": "Spizelloides arborea",
        "Cape-Glossy Starling": "Lamprotornis nitens",
        "Artic Tern": "Sterna paradisaea",
        "Myrtle Warbler": "Setophaga coronata",
        "Swainson Warbler": "Limnothlypis swainsonii",
        "Wilson Warbler": "Cardellina pusilla",
        "Yellow Warbler": "Setophaga aestiva",
        "Bewick Wren": "Thryomanes bewickii",
        "House Wren": "Troglodytes aedon",
    }
)

# CUB predates several modern species splits.  A single canonical match is
# useful for text labels, but strict leakage prevention must exclude every
# plausible descendant of the historical label from auxiliary training.
_ADDITIONAL_EXCLUSION_SCIENTIFIC_NAMES: Mapping[str, tuple[str, ...]] = (
    MappingProxyType(
        {
            "Herring Gull": ("Larus smithsonianus",),
            "Green Violetear": ("Colibri cyanotus",),
            "Yellow Warbler": ("Setophaga petechia",),
            "House Wren": ("Troglodytes musculus",),
        }
    )
)

# ``Sayornis`` is a genus label, not a species label.  CUB provides no safe
# basis for choosing Black Phoebe, Eastern Phoebe, or Say's Phoebe, so all
# BirdNET species in the genus are exclusion targets and none is canonical.
_AMBIGUOUS_CUB_GENERA: Mapping[str, str] = MappingProxyType({"Sayornis": "Sayornis"})

CrosswalkStatus = Literal["exact_alias", "scientific_override", "ambiguous_genus"]


@dataclass(frozen=True, slots=True)
class CubBaseNovelSplit:
    """The deterministic 100/100 class-disjoint CUB split."""

    protocol: str
    salt: str
    class_names: tuple[str, ...]
    base_class_ids: tuple[int, ...]
    novel_class_ids: tuple[int, ...]
    digest: str

    def partition_for(self, class_id: int) -> Literal["base", "novel"]:
        """Return the partition for a zero-based CUB class id."""

        if not 0 <= class_id < len(self.class_names):
            raise IndexError(f"CUB class id is out of range: {class_id}")
        return "base" if class_id in self.base_class_ids else "novel"


@dataclass(frozen=True, slots=True)
class CubBirdNetTaxonMatch:
    """One CUB class and its conservative BirdNET alignment."""

    cub_class_id: int
    cub_class_name: str
    status: CrosswalkStatus
    birdnet_id: str | None
    scientific_name: str | None
    birdnet_common_name: str | None
    matched_alias: str | None
    exclusion_birdnet_ids: tuple[str, ...]
    exclusion_scientific_names: tuple[str, ...]
    note: str


@dataclass(frozen=True, slots=True)
class CubBirdNetCrosswalk:
    """Class-order CUB-to-BirdNET mapping locked to one source CSV."""

    protocol: str
    entries: tuple[CubBirdNetTaxonMatch, ...]
    birdnet_csv_sha256: str
    digest: str

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(entry.cub_class_name for entry in self.entries)

    @property
    def taxon_ids(self) -> tuple[str | None, ...]:
        """Canonical BirdNET ids in CUB class order; ambiguous classes are ``None``."""

        return tuple(entry.birdnet_id for entry in self.entries)

    @property
    def status_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(entry.status for entry in self.entries).items()))

    def entry_for_class_id(self, class_id: int) -> CubBirdNetTaxonMatch:
        if not 0 <= class_id < len(self.entries):
            raise IndexError(f"CUB class id is out of range: {class_id}")
        return self.entries[class_id]

    def excluded_birdnet_ids(self, class_ids: Iterable[int]) -> tuple[str, ...]:
        """Return the strict union of BirdNET ids excluded for CUB classes."""

        result: set[str] = set()
        for class_id in class_ids:
            result.update(self.entry_for_class_id(class_id).exclusion_birdnet_ids)
        return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class _BirdNetRow:
    birdnet_id: str
    scientific_name: str
    common_name: str
    common_aliases: tuple[str, ...]


def _canonical_json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_alias(value: str) -> str:
    """Normalize only case, accents, whitespace, and punctuation."""

    decomposed = unicodedata.normalize("NFKD", value).casefold()
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return "".join(character for character in without_marks if character.isalnum())


def read_cub_class_names(path: Path | str) -> tuple[str, ...]:
    """Read and validate the canonical ordered FuDD/CUB class-name JSON."""

    source = Path(path).expanduser()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != _CUB_CLASS_COUNT:
        raise ValueError(f"Expected a JSON list of {_CUB_CLASS_COUNT} CUB class names")
    if any(not isinstance(name, str) or not name.strip() for name in value):
        raise ValueError("Every CUB class name must be a non-empty string")
    names = tuple(name.strip() for name in value)
    normalized = tuple(_normalise_alias(name) for name in names)
    if len(set(normalized)) != len(normalized):
        raise ValueError("CUB class names must be unique after normalization")
    return names


def make_strict_cub_base_novel_split(
    class_names: Sequence[str],
    *,
    salt: str = CUB_BASE_NOVEL_SPLIT_SALT,
) -> CubBaseNovelSplit:
    """Choose one base and one novel class per adjacent pair with salted SHA-256."""

    names = tuple(class_names)
    if len(names) != _CUB_CLASS_COUNT:
        raise ValueError(f"Expected {_CUB_CLASS_COUNT} CUB class names")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Every CUB class name must be a non-empty string")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")
    base: list[int] = []
    novel: list[int] = []
    for first_id in range(0, _CUB_CLASS_COUNT, 2):
        pair = (first_id, first_id + 1)
        ranked = sorted(
            pair,
            key=lambda class_id: (
                hashlib.sha256(
                    f"{salt}\0{class_id}\0{names[class_id]}".encode()
                ).digest(),
                class_id,
            ),
        )
        base.append(ranked[0])
        novel.append(ranked[1])
    base_ids = tuple(base)
    novel_ids = tuple(novel)
    base_set = set(base_ids)
    assignments = [
        {
            "class_id": class_id,
            "class_name": name,
            "partition": "base" if class_id in base_set else "novel",
        }
        for class_id, name in enumerate(names)
    ]
    digest = _canonical_json_digest(
        {
            "protocol": CUB_BASE_NOVEL_PROTOCOL,
            "salt": salt,
            "assignments": assignments,
        }
    )
    return CubBaseNovelSplit(
        protocol=CUB_BASE_NOVEL_PROTOCOL,
        salt=salt,
        class_names=names,
        base_class_ids=base_ids,
        novel_class_ids=novel_ids,
        digest=digest,
    )


def load_strict_cub_base_novel_split(class_names_json: Path | str) -> CubBaseNovelSplit:
    """Read the canonical class order and construct the strict split."""

    return make_strict_cub_base_novel_split(read_cub_class_names(class_names_json))


def _split_aliases(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split("|") if part.strip())


def _read_birdnet_species(path: Path) -> tuple[_BirdNetRow, ...]:
    required = {
        "birdnet_id",
        "scientific_name",
        "common_name",
        "common_name_alt",
        "taxon_group",
        "record_type",
        "scientific_name_aliases",
        "common_name_aliases",
        "common_name_en",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"BirdNET CSV is missing columns: {sorted(missing)}")
        rows: list[_BirdNetRow] = []
        seen_ids: set[str] = set()
        for line_number, raw in enumerate(reader, start=2):
            if raw["taxon_group"].strip() != "Aves" or raw["record_type"].strip() != "species":
                continue
            birdnet_id = raw["birdnet_id"].strip()
            scientific_name = raw["scientific_name"].strip()
            common_name = raw["common_name"].strip()
            if not birdnet_id or not scientific_name or not common_name:
                raise ValueError(f"Incomplete BirdNET species row at line {line_number}")
            if birdnet_id in seen_ids:
                raise ValueError(f"Duplicate BirdNET id at line {line_number}: {birdnet_id}")
            seen_ids.add(birdnet_id)
            aliases: list[str] = []
            for column in (
                "common_name",
                "common_name_alt",
                "common_name_aliases",
                "common_name_en",
            ):
                aliases.extend(_split_aliases(raw[column]))
            rows.append(
                _BirdNetRow(
                    birdnet_id=birdnet_id,
                    scientific_name=scientific_name,
                    common_name=common_name,
                    common_aliases=tuple(dict.fromkeys(aliases)),
                )
            )
    if not rows:
        raise ValueError("BirdNET CSV contains no Aves species rows")
    return tuple(rows)


def _unique_row_for_scientific_name(
    scientific_name: str,
    rows_by_scientific_name: Mapping[str, tuple[_BirdNetRow, ...]],
) -> _BirdNetRow:
    matches = rows_by_scientific_name.get(scientific_name, ())
    if len(matches) != 1:
        raise ValueError(
            "Reviewed scientific name must resolve to exactly one BirdNET species: "
            f"{scientific_name!r} resolved to {len(matches)}"
        )
    return matches[0]


def build_cub_birdnet_crosswalk(
    class_names_json: Path | str,
    birdnet_csv: Path | str,
) -> CubBirdNetCrosswalk:
    """Build a fail-closed CUB-to-BirdNET crosswalk from runtime source files."""

    class_names = read_cub_class_names(class_names_json)
    csv_path = Path(birdnet_csv).expanduser()
    rows = _read_birdnet_species(csv_path)

    aliases: dict[str, list[tuple[_BirdNetRow, str]]] = {}
    scientific: dict[str, list[_BirdNetRow]] = {}
    for row in rows:
        scientific.setdefault(row.scientific_name, []).append(row)
        for alias in row.common_aliases:
            aliases.setdefault(_normalise_alias(alias), []).append((row, alias))
    rows_by_scientific = {key: tuple(value) for key, value in scientific.items()}

    entries: list[CubBirdNetTaxonMatch] = []
    for class_id, class_name in enumerate(class_names):
        ambiguous_genus = _AMBIGUOUS_CUB_GENERA.get(class_name)
        if ambiguous_genus is not None:
            genus_rows = tuple(
                sorted(
                    (
                        row
                        for row in rows
                        if row.scientific_name.partition(" ")[0] == ambiguous_genus
                    ),
                    key=lambda row: row.birdnet_id,
                )
            )
            if len(genus_rows) < 2:
                raise ValueError(
                    f"Ambiguous genus {ambiguous_genus!r} did not resolve to multiple species"
                )
            entries.append(
                CubBirdNetTaxonMatch(
                    cub_class_id=class_id,
                    cub_class_name=class_name,
                    status="ambiguous_genus",
                    birdnet_id=None,
                    scientific_name=None,
                    birdnet_common_name=None,
                    matched_alias=None,
                    exclusion_birdnet_ids=tuple(row.birdnet_id for row in genus_rows),
                    exclusion_scientific_names=tuple(
                        row.scientific_name for row in genus_rows
                    ),
                    note=(
                        "CUB uses the genus-level label Sayornis; no species is asserted and "
                        "every BirdNET species in that genus is excluded from training."
                    ),
                )
            )
            continue

        override = CUB_SCIENTIFIC_NAME_OVERRIDES.get(class_name)
        if override is not None:
            canonical = _unique_row_for_scientific_name(override, rows_by_scientific)
            exclusion_rows = [canonical]
            for additional in _ADDITIONAL_EXCLUSION_SCIENTIFIC_NAMES.get(class_name, ()):
                exclusion_rows.append(
                    _unique_row_for_scientific_name(additional, rows_by_scientific)
                )
            exclusion_rows.sort(key=lambda row: row.birdnet_id)
            entries.append(
                CubBirdNetTaxonMatch(
                    cub_class_id=class_id,
                    cub_class_name=class_name,
                    status="scientific_override",
                    birdnet_id=canonical.birdnet_id,
                    scientific_name=canonical.scientific_name,
                    birdnet_common_name=canonical.common_name,
                    matched_alias=None,
                    exclusion_birdnet_ids=tuple(row.birdnet_id for row in exclusion_rows),
                    exclusion_scientific_names=tuple(
                        row.scientific_name for row in exclusion_rows
                    ),
                    note=(
                        "Reviewed scientific-name override for a possessive, legacy, spelling, "
                        "or generic CUB label; exclusions include plausible modern split taxa."
                    ),
                )
            )
            continue

        candidates = aliases.get(_normalise_alias(class_name), [])
        unique_candidates = {
            candidate.birdnet_id: (candidate, alias) for candidate, alias in candidates
        }
        if len(unique_candidates) != 1:
            ids = sorted(unique_candidates)
            raise ValueError(
                f"CUB label {class_name!r} must match one BirdNET alias, found {ids}"
            )
        canonical, matched_alias = next(iter(unique_candidates.values()))
        entries.append(
            CubBirdNetTaxonMatch(
                cub_class_id=class_id,
                cub_class_name=class_name,
                status="exact_alias",
                birdnet_id=canonical.birdnet_id,
                scientific_name=canonical.scientific_name,
                birdnet_common_name=canonical.common_name,
                matched_alias=matched_alias,
                exclusion_birdnet_ids=(canonical.birdnet_id,),
                exclusion_scientific_names=(canonical.scientific_name,),
                note="Unique normalized exact English alias match.",
            )
        )

    if len(entries) != _CUB_CLASS_COUNT:
        raise AssertionError("Internal error: crosswalk is not in complete CUB class order")
    csv_digest = _sha256_file(csv_path)
    digest_payload = {
        "protocol": CUB_BIRDNET_CROSSWALK_PROTOCOL,
        "birdnet_csv_sha256": csv_digest,
        "entries": [asdict(entry) for entry in entries],
    }
    return CubBirdNetCrosswalk(
        protocol=CUB_BIRDNET_CROSSWALK_PROTOCOL,
        entries=tuple(entries),
        birdnet_csv_sha256=csv_digest,
        digest=_canonical_json_digest(digest_payload),
    )
