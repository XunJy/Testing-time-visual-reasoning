from __future__ import annotations

from ttvr.methods.fudd.prompts import CubPromptRepository


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
