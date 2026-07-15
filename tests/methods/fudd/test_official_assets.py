from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from ttvr.methods.fudd.prompts import load_official_prompts

ASSET_ROOT = Path(__file__).resolve().parents[3] / "data" / "fudd_official"
ASSET_FILES = ("cub_class_names.json", "cub_prompt_pairs.json")

pytestmark = pytest.mark.skipif(
    any(not (ASSET_ROOT / filename).is_file() for filename in ASSET_FILES),
    reason=(
        "Official FuDD prompt assets are downloaded at runtime and are not "
        "redistributed in the public repository"
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("filename", "size", "expected_sha256"),
    [
        (
            "cub_class_names.json",
            4_334,
            "01eff4094773d49b92210f56efb9cab13860c0cdaf9953ff1e0ef082cf226723",
        ),
        (
            "cub_prompt_pairs.json",
            21_985_463,
            "cc4300eaaf7c7bf46e515839ebe03abe827e86dafc3f49f63ea97a6c4c035237",
        ),
    ],
)
def test_official_asset_matches_recorded_upstream_copy(
    filename: str,
    size: int,
    expected_sha256: str,
) -> None:
    path = ASSET_ROOT / filename

    assert path.stat().st_size == size
    assert _sha256(path) == expected_sha256


def test_official_cub_prompt_cache_is_complete_and_well_formed() -> None:
    class_names = json.loads((ASSET_ROOT / "cub_class_names.json").read_text())
    prompt_pairs = json.loads((ASSET_ROOT / "cub_prompt_pairs.json").read_text())

    assert len(class_names) == 200
    assert len(prompt_pairs) == math.comb(len(class_names), 2)

    for pair_key, pair_data in prompt_pairs.items():
        class_a, class_b = pair_data["classes"]
        assert 0 <= class_a < class_b < len(class_names)
        assert pair_key == f"{class_a}_{class_b}"
        assert 3 <= len(pair_data["prompt_pairs"]) <= 5
        for description in pair_data["prompt_pairs"]:
            assert description["attr_type"]
            assert len(description["prompt_pair"]) == 2
            assert all(prompt.strip() for prompt in description["prompt_pair"])


def test_official_assets_load_with_global_class_alignment() -> None:
    repository = load_official_prompts(ASSET_ROOT)

    assert repository.class_count == 200
    assert repository.pair_count == math.comb(200, 2)
    assert repository.class_names[:2] == (
        "Black-footed Albatross",
        "Laysan Albatross",
    )
    forward = repository.pair_prompts(0, 1)
    reverse = repository.pair_prompts(1, 0)
    assert reverse == (forward[1], forward[0])
    assert forward[0][0] == (
        "A photo of a black-footed albatross, a type of bird, with a yellow bill."
    )
