from __future__ import annotations

from types import MappingProxyType

import pytest

from ttvr.methods.fudd.prompts import (
    ClassPairDescriptions,
    CubPromptRepository,
    DifferentialDescription,
)


@pytest.fixture
def tiny_prompt_repository() -> CubPromptRepository:
    """Create an immutable three-class repository for prompt logic tests."""

    pairs = {
        (0, 1): ClassPairDescriptions(
            first_id=0,
            second_id=1,
            descriptions=(
                DifferentialDescription(
                    attribute="color",
                    first="Alpha bird is red.",
                    second="Beta bird is blue.",
                ),
                DifferentialDescription(
                    attribute="size",
                    first="Alpha bird is shared.",
                    second="Beta bird is small.",
                ),
            ),
        ),
        (0, 2): ClassPairDescriptions(
            first_id=0,
            second_id=2,
            descriptions=(
                DifferentialDescription(
                    attribute="shape",
                    first="Alpha bird is shared.",
                    second="Gamma bird is round.",
                ),
                DifferentialDescription(
                    attribute="bill",
                    first="Alpha bird has a hook.",
                    second="Gamma bird has a straight bill.",
                ),
            ),
        ),
        (1, 2): ClassPairDescriptions(
            first_id=1,
            second_id=2,
            descriptions=(
                DifferentialDescription(
                    attribute="wing",
                    first="Beta bird is striped.",
                    second="Gamma bird is plain.",
                ),
            ),
        ),
    }
    return CubPromptRepository(
        class_names=("Alpha bird", "Beta bird", "Gamma bird"),
        pairs=MappingProxyType(pairs),
        source_digest="synthetic-test-fixture",
    )
