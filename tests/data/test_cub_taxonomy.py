from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ttvr.data.cub_taxonomy import (
    CUB_BASE_NOVEL_SPLIT_SALT,
    CUB_SCIENTIFIC_NAME_OVERRIDES,
    build_cub_birdnet_crosswalk,
    load_strict_cub_base_novel_split,
    make_strict_cub_base_novel_split,
    read_cub_class_names,
)
from ttvr.data.inat2021 import _birdnet_indices, _resolve_taxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES_JSON = PROJECT_ROOT / "data/fudd_official/cub_class_names.json"

_CSV_COLUMNS = (
    "birdnet_id",
    "scientific_name",
    "common_name",
    "common_name_alt",
    "taxon_group",
    "record_type",
    "scientific_name_aliases",
    "common_name_aliases",
    "common_name_en",
)

_ADDITIONAL_SPLIT_TAXA = (
    ("Larus smithsonianus", "American Herring Gull"),
    ("Colibri cyanotus", "Lesser Violetear"),
    ("Setophaga petechia", "Mangrove Warbler"),
    ("Troglodytes musculus", "Southern House Wren"),
)


def _row(
    birdnet_id: str,
    scientific_name: str,
    common_name: str,
    *,
    common_name_alt: str = "",
    taxon_group: str = "Aves",
) -> dict[str, str]:
    return {
        "birdnet_id": birdnet_id,
        "scientific_name": scientific_name,
        "common_name": common_name,
        "common_name_alt": common_name_alt,
        "taxon_group": taxon_group,
        "record_type": "species",
        "scientific_name_aliases": "",
        "common_name_aliases": "",
        "common_name_en": common_name,
    }


def _write_complete_synthetic_birdnet(
    path: Path,
    *,
    duplicate_blue_jay_alias: bool = False,
) -> None:
    class_names = read_cub_class_names(CLASS_NAMES_JSON)
    rows: list[dict[str, str]] = []
    for class_id, class_name in enumerate(class_names):
        if class_name == "Sayornis":
            continue
        scientific_name = (
            "Corvus brachyrhynchos"
            if class_name == "American Crow"
            else CUB_SCIENTIFIC_NAME_OVERRIDES.get(class_name, f"Syntheticus{class_id} example")
        )
        common_name = (
            f"Reviewed canonical name {class_id}"
            if class_name in CUB_SCIENTIFIC_NAME_OVERRIDES
            else class_name
        )
        # Exercise punctuation-insensitive exact alias matching without
        # changing the synthetic class identity.
        common_name_alt = "Black footed Albatross" if class_id == 0 else ""
        rows.append(
            _row(
                "BN03722" if class_name == "American Crow" else f"BN{class_id:05d}",
                scientific_name,
                common_name,
                common_name_alt=common_name_alt,
            )
        )
    for offset, (scientific_name, common_name) in enumerate(_ADDITIONAL_SPLIT_TAXA):
        rows.append(_row(f"BNS{offset:04d}", scientific_name, common_name))
    for offset, (scientific_name, common_name) in enumerate(
        (
            ("Sayornis nigricans", "Black Phoebe"),
            ("Sayornis phoebe", "Eastern Phoebe"),
            ("Sayornis saya", "Say's Phoebe"),
        )
    ):
        rows.append(_row(f"BNG{offset:04d}", scientific_name, common_name))
    # Non-bird rows must not participate in alias resolution.
    rows.append(_row("INSECT1", "Synthetic moth", "Blue Jay", taxon_group="Insecta"))
    if duplicate_blue_jay_alias:
        rows.append(_row("DUPLICATE", "Duplicateus jay", "Blue-Jay"))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_strict_split_is_balanced_pairwise_deterministic_and_digest_locked() -> None:
    split = load_strict_cub_base_novel_split(CLASS_NAMES_JSON)

    assert split.salt == CUB_BASE_NOVEL_SPLIT_SALT
    assert len(split.base_class_ids) == 100
    assert len(split.novel_class_ids) == 100
    assert set(split.base_class_ids).isdisjoint(split.novel_class_ids)
    assert set(split.base_class_ids) | set(split.novel_class_ids) == set(range(200))
    for first_id in range(0, 200, 2):
        pair = {first_id, first_id + 1}
        assert len(pair & set(split.base_class_ids)) == 1
        assert len(pair & set(split.novel_class_ids)) == 1
    assert split == load_strict_cub_base_novel_split(CLASS_NAMES_JSON)
    assert split.digest == "6f8ad44f74a748b901ef575855b20974474a0ad007a053e5f616dc9fcd404405"


def test_split_salt_changes_assignments_and_digest() -> None:
    names = read_cub_class_names(CLASS_NAMES_JSON)
    first = make_strict_cub_base_novel_split(names, salt="salt-one")
    second = make_strict_cub_base_novel_split(names, salt="salt-two")

    assert first.base_class_ids != second.base_class_ids
    assert first.digest != second.digest


def test_crosswalk_is_in_class_order_and_marks_sayornis_ambiguous(tmp_path: Path) -> None:
    csv_path = tmp_path / "birdnet.csv"
    _write_complete_synthetic_birdnet(csv_path)

    crosswalk = build_cub_birdnet_crosswalk(CLASS_NAMES_JSON, csv_path)

    assert len(crosswalk.entries) == 200
    assert crosswalk.class_names == read_cub_class_names(CLASS_NAMES_JSON)
    assert len(crosswalk.taxon_ids) == 200
    assert crosswalk.status_counts == {
        "ambiguous_genus": 1,
        "exact_alias": 165,
        "scientific_override": 34,
    }
    assert crosswalk.entries[0].status == "exact_alias"
    assert crosswalk.entries[8].status == "scientific_override"
    assert crosswalk.entries[8].scientific_name == "Euphagus cyanocephalus"

    sayornis = crosswalk.entries[102]
    assert sayornis.cub_class_name == "Sayornis"
    assert sayornis.status == "ambiguous_genus"
    assert sayornis.birdnet_id is None
    assert crosswalk.taxon_ids[102] is None
    assert sayornis.exclusion_scientific_names == (
        "Sayornis nigricans",
        "Sayornis phoebe",
        "Sayornis saya",
    )
    assert crosswalk.excluded_birdnet_ids((102,)) == (
        "BNG0000",
        "BNG0001",
        "BNG0002",
    )
    assert "genus-level" in sayornis.note
    assert crosswalk == build_cub_birdnet_crosswalk(CLASS_NAMES_JSON, csv_path)


def test_crosswalk_conservatively_excludes_modern_split_taxa(tmp_path: Path) -> None:
    csv_path = tmp_path / "birdnet.csv"
    _write_complete_synthetic_birdnet(csv_path)
    crosswalk = build_cub_birdnet_crosswalk(CLASS_NAMES_JSON, csv_path)

    herring_gull = crosswalk.entries[61]
    assert herring_gull.exclusion_scientific_names == (
        "Larus argentatus",
        "Larus smithsonianus",
    )
    green_violetear = crosswalk.entries[69]
    assert set(green_violetear.exclusion_scientific_names) == {
        "Colibri cyanotus",
        "Colibri thalassinus",
    }
    yellow_warbler = crosswalk.entries[181]
    assert set(yellow_warbler.exclusion_scientific_names) == {
        "Setophaga aestiva",
        "Setophaga petechia",
    }
    house_wren = crosswalk.entries[195]
    assert set(house_wren.exclusion_scientific_names) == {
        "Troglodytes aedon",
        "Troglodytes musculus",
    }


def test_inat_northwestern_crow_resolves_into_cub_american_crow_exclusion(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "birdnet.csv"
    _write_complete_synthetic_birdnet(csv_path)
    scientific, common, rows = _birdnet_indices(csv_path)
    source_taxon = _resolve_taxon(
        {
            "id": 3750,
            "name": "Corvus caurinus",
            "common_name": "Northwestern Crow",
        },
        scientific,
        common,
        rows,
    )
    crosswalk = build_cub_birdnet_crosswalk(CLASS_NAMES_JSON, csv_path)
    excluded_taxa = {
        f"birdnet:{birdnet_id}" for birdnet_id in crosswalk.excluded_birdnet_ids(range(200))
    }

    assert source_taxon.taxon_id == "birdnet:BN03722"
    assert crosswalk.entries[28].cub_class_name == "American Crow"
    assert source_taxon.taxon_id in excluded_taxa


def test_crosswalk_rejects_an_alias_collision_instead_of_guessing(tmp_path: Path) -> None:
    csv_path = tmp_path / "birdnet.csv"
    _write_complete_synthetic_birdnet(csv_path, duplicate_blue_jay_alias=True)

    with pytest.raises(ValueError, match="Blue Jay"):
        build_cub_birdnet_crosswalk(CLASS_NAMES_JSON, csv_path)
