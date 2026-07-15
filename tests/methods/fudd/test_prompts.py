from __future__ import annotations

import json
from pathlib import Path

import pytest

from ttvr.methods.fudd.prompts import (
    ClassPairDescriptions,
    CubPromptRepository,
    DifferentialDescription,
    load_pair_overrides,
)


def test_single_template_prompts_follow_class_order(
    tiny_prompt_repository: CubPromptRepository,
) -> None:
    repository = tiny_prompt_repository
    assert repository.class_names == ("Alpha bird", "Beta bird", "Gamma bird")
    assert repository.pair_count == 3
    assert repository.single_template_prompts() == (
        ("A photo of a alpha bird.",),
        ("A photo of a beta bird.",),
        ("A photo of a gamma bird.",),
    )


def test_pair_prompts_follow_requested_class_order(
    tiny_prompt_repository: CubPromptRepository,
) -> None:
    repository = tiny_prompt_repository
    forward = repository.pair_prompts(0, 2)
    reverse = repository.pair_prompts(2, 0)

    assert forward == (
        ("Alpha bird is shared.", "Alpha bird has a hook."),
        ("Gamma bird is round.", "Gamma bird has a straight bill."),
    )
    assert reverse == (forward[1], forward[0])


def test_candidate_prompt_aggregation_preserves_candidate_order_and_stable_dedup(
    tiny_prompt_repository: CubPromptRepository,
) -> None:
    repository = tiny_prompt_repository
    prompts = repository.prompts_for_candidates([2, 0, 1])

    assert prompts == (
        (
            "Gamma bird is round.",
            "Gamma bird has a straight bill.",
            "Gamma bird is plain.",
        ),
        (
            "Alpha bird is shared.",
            "Alpha bird has a hook.",
            "Alpha bird is red.",
        ),
        (
            "Beta bird is striped.",
            "Beta bird is blue.",
            "Beta bird is small.",
        ),
    )
    assert prompts[1].count("Alpha bird is shared.") == 1


def _replacement_pair(*, suffix: str = "") -> ClassPairDescriptions:
    return ClassPairDescriptions(
        first_id=0,
        second_id=2,
        descriptions=(
            DifferentialDescription(
                attribute="wing",
                first=f"Alpha bird has a pale wing{suffix}.",
                second="Gamma bird has a dark wing.",
            ),
        ),
    )


def test_pair_overrides_are_immutable_isolated_and_digest_namespaced(
    tiny_prompt_repository: CubPromptRepository,
) -> None:
    replacement = _replacement_pair()

    first = tiny_prompt_repository.with_pair_overrides({(0, 2): replacement})
    second = tiny_prompt_repository.with_pair_overrides({(0, 2): replacement})
    changed = tiny_prompt_repository.with_pair_overrides(
        {(0, 2): _replacement_pair(suffix=" edge")}
    )

    assert tiny_prompt_repository.pairs[(0, 2)] != replacement
    assert first.pairs[(0, 2)] == replacement
    assert first.pair_count == tiny_prompt_repository.pair_count
    assert first.source_digest == second.source_digest
    assert first.source_digest != tiny_prompt_repository.source_digest
    assert first.source_digest != changed.source_digest
    with pytest.raises(TypeError):
        first.pairs[(0, 1)] = replacement  # type: ignore[index]


def test_pair_overrides_reject_mismatched_key(
    tiny_prompt_repository: CubPromptRepository,
) -> None:
    with pytest.raises(ValueError, match="does not match"):
        tiny_prompt_repository.with_pair_overrides({(0, 1): _replacement_pair()})


def test_load_pair_overrides_uses_official_pair_schema(tmp_path: Path) -> None:
    path = tmp_path / "override.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pairs": {
                    "0_2": {
                        "classes": [0, 2],
                        "prompt_pairs": [
                            {
                                "attr_type": "wing",
                                "prompt_pair": [
                                    "Alpha bird has a pale wing.",
                                    "Gamma bird has a dark wing.",
                                ],
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    overrides = load_pair_overrides(path)

    assert tuple(overrides) == ((0, 2),)
    assert overrides[(0, 2)].descriptions[0].attribute == "wing"
