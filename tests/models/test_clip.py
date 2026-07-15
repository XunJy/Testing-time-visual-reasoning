from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as functional

from ttvr.methods.fudd.evaluation import _batched_fudd_predictions
from ttvr.methods.fudd.prompts import CubPromptRepository
from ttvr.metrics import ordered_predictions
from ttvr.models import CLIPBackend, TextFeatureTable


class _FakeClipModel:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.moved_to: torch.device | None = None

    def parameters(self):
        return iter((self.parameter,))

    def to(self, device):
        self.moved_to = torch.device(device)
        return self

    def eval(self):
        return self


def test_official_precision_loads_on_cpu_before_moving_to_cuda(monkeypatch) -> None:
    calls: list[object] = []

    def fake_load(model_name, *, device, jit, download_root):
        calls.append(device)
        return _FakeClipModel(), object()

    monkeypatch.setitem(sys.modules, "clip", SimpleNamespace(load=fake_load))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    official = CLIPBackend(device="cuda", precision="fp32")
    native = CLIPBackend(device="cuda", precision="native")

    assert calls == ["cpu", torch.device("cuda")]
    assert official.model.moved_to == torch.device("cuda")
    assert native.model.moved_to is None
    assert official.feature_dtype_name == "torch.float32"


def test_prompt_pooling_matches_official_normalise_mean_normalise() -> None:
    backend = object.__new__(CLIPBackend)
    texts = ("a", "b", "c", "d")
    features = functional.normalize(
        torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [2.0, 0.0, 1.0],
                [0.0, 1.0, 2.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        dim=-1,
    )
    table = TextFeatureTable(
        texts=texts,
        features=features,
        index={text: index for index, text in enumerate(texts)},
    )
    groups = (("a", "c", "d"), ("b", "a"))

    actual = backend.pool_prompt_groups(groups, table)
    expected = torch.stack(
        [
            functional.normalize(features[[0, 2, 3]].mean(dim=0), dim=0),
            functional.normalize(features[[1, 0]].mean(dim=0), dim=0),
        ]
    )

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_batched_fudd_matches_per_image_reference_with_reordered_candidates(
    tiny_prompt_repository: CubPromptRepository,
) -> None:
    backend = object.__new__(CLIPBackend)
    backend.device = torch.device("cpu")
    candidate_rows = [[2, 0, 1], [1, 2, 0]]
    candidates = torch.tensor(candidate_rows)
    required_texts = tuple(
        dict.fromkeys(
            text
            for row in candidate_rows
            for group in tiny_prompt_repository.prompts_for_candidates(row)
            for text in group
        )
    )
    generator = torch.Generator().manual_seed(2026)
    text_features = functional.normalize(
        torch.randn(len(required_texts), 6, generator=generator),
        dim=-1,
    )
    image_features = functional.normalize(
        torch.randn(2, 6, generator=generator),
        dim=-1,
    )
    table = TextFeatureTable(
        texts=required_texts,
        features=text_features,
        index={text: index for index, text in enumerate(required_texts)},
    )

    actual = _batched_fudd_predictions(
        image_features,
        candidates,
        candidate_rows,
        tiny_prompt_repository,
        backend,
        table,
        batch_size=2,
        progress=None,
    )

    expected_rows = []
    for image, candidate_row in zip(image_features, candidate_rows, strict=True):
        prototypes = []
        for group in tiny_prompt_repository.prompts_for_candidates(candidate_row):
            rows = torch.stack([text_features[table.index[text]] for text in group])
            prototypes.append(functional.normalize(rows.mean(dim=0), dim=0))
        logits = image @ torch.stack(prototypes).t()
        expected_rows.append(ordered_predictions(logits, torch.tensor(candidate_row)))

    assert torch.equal(actual, torch.stack(expected_rows))
