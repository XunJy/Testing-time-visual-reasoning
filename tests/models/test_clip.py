from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

from ttvr.methods.fudd.evaluation import _batched_fudd_predictions, rerank_candidates
from ttvr.methods.fudd.prompts import CubPromptRepository
from ttvr.metrics import ordered_predictions
from ttvr.models import CLIPBackend, TextFeatureTable
from ttvr.models.clip import (
    OPENAI_CLIP_COMMIT,
    OPENAI_CLIP_REPOSITORY,
    VIT_L14_336_CHECKPOINT_FILENAME,
    VIT_L14_336_CHECKPOINT_SHA256,
    OpenAIClipCheckpoint,
    OpenAIClipInstallation,
    verify_openai_clip_checkpoint,
    verify_openai_clip_installation,
)


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


class _FakeClipDistribution:
    version = "1.0"
    metadata = {"Name": "clip"}

    def __init__(self, direct_url: dict[str, object]) -> None:
        self.direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return json.dumps(self.direct_url)


def test_openai_clip_installation_requires_exact_pep610_commit(monkeypatch) -> None:
    direct_url = {
        "url": OPENAI_CLIP_REPOSITORY,
        "vcs_info": {"vcs": "git", "commit_id": OPENAI_CLIP_COMMIT},
    }
    monkeypatch.setattr(
        "ttvr.models.clip.importlib.metadata.distribution",
        lambda name: _FakeClipDistribution(direct_url),
    )

    identity = verify_openai_clip_installation(expected_commit=OPENAI_CLIP_COMMIT)
    assert identity.commit_id == OPENAI_CLIP_COMMIT
    assert identity.repository_url == OPENAI_CLIP_REPOSITORY

    direct_url["vcs_info"]["commit_id"] = "0" * 40
    with pytest.raises(RuntimeError, match="commit mismatch"):
        verify_openai_clip_installation(expected_commit=OPENAI_CLIP_COMMIT)


def test_openai_clip_checkpoint_requires_exact_sha256(tmp_path: Path) -> None:
    payload = b"locked OpenAI CLIP weights"
    checkpoint = tmp_path / "ViT-L-14-336px.pt"
    checkpoint.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    identity = verify_openai_clip_checkpoint(
        tmp_path,
        model_name="ViT-L/14@336px",
        checkpoint_filename=checkpoint.name,
        expected_sha256=digest,
    )
    assert identity.sha256 == digest
    assert identity.size_bytes == len(payload)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_openai_clip_checkpoint(
            tmp_path,
            model_name="ViT-L/14@336px",
            checkpoint_filename=checkpoint.name,
            expected_sha256="0" * 64,
        )


def _verified_installation(commit_id: str = OPENAI_CLIP_COMMIT) -> OpenAIClipInstallation:
    return OpenAIClipInstallation(
        distribution="clip",
        version="1.0",
        repository_url=OPENAI_CLIP_REPOSITORY,
        vcs="git",
        commit_id=commit_id,
    )


def test_official_precision_loads_on_cpu_before_moving_to_cuda(monkeypatch) -> None:
    calls: list[object] = []

    def fake_load(model_name, *, device, jit, download_root):
        calls.append(device)
        return _FakeClipModel(), object()

    monkeypatch.setitem(sys.modules, "clip", SimpleNamespace(load=fake_load))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "ttvr.models.clip.verify_openai_clip_installation",
        lambda **kwargs: _verified_installation(),
    )

    official = CLIPBackend(device="cuda", precision="fp32")
    native = CLIPBackend(device="cuda", precision="native")

    assert calls == ["cpu", torch.device("cuda")]
    assert official.model.moved_to == torch.device("cuda")
    assert native.model.moved_to is None
    assert official.feature_dtype_name == "torch.float32"


def test_backend_identity_comes_from_verified_runtime_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_commit = "b" * 40
    calls: dict[str, object] = {}

    def fake_load(model_name, *, device, jit, download_root):
        calls["load"] = (model_name, device, jit, download_root)
        return _FakeClipModel(), object()

    def verify_installation(**kwargs: object) -> OpenAIClipInstallation:
        calls["installation"] = kwargs
        return _verified_installation(actual_commit)

    def verify_checkpoint(*args: object, **kwargs: object) -> OpenAIClipCheckpoint:
        calls["checkpoint"] = (args, kwargs)
        return OpenAIClipCheckpoint(
            model_name="ViT-L/14@336px",
            path=tmp_path / VIT_L14_336_CHECKPOINT_FILENAME,
            sha256=VIT_L14_336_CHECKPOINT_SHA256,
            size_bytes=123,
        )

    monkeypatch.setitem(sys.modules, "clip", SimpleNamespace(load=fake_load))
    monkeypatch.setattr(
        "ttvr.models.clip.verify_openai_clip_installation",
        verify_installation,
    )
    monkeypatch.setattr(
        "ttvr.models.clip.verify_openai_clip_checkpoint",
        verify_checkpoint,
    )

    backend = CLIPBackend(device="cpu", model_cache_dir=tmp_path)

    assert calls["installation"] == {"expected_commit": OPENAI_CLIP_COMMIT}
    assert calls["checkpoint"] == (
        (tmp_path,),
        {
            "model_name": "ViT-L/14@336px",
            "checkpoint_filename": VIT_L14_336_CHECKPOINT_FILENAME,
            "expected_sha256": VIT_L14_336_CHECKPOINT_SHA256,
        },
    )
    assert backend.cache_identity == f"openai-clip:ViT-L/14@336px@{actual_commit}"
    assert backend.openai_clip_installation.commit_id == actual_commit
    assert backend.openai_clip_checkpoint is not None
    assert backend.openai_clip_checkpoint.sha256 == VIT_L14_336_CHECKPOINT_SHA256


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
    result = rerank_candidates(
        image_features,
        candidates,
        tiny_prompt_repository,
        backend,
        text_table=table,
        batch_size=1,
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
    assert torch.equal(result.ranked_class_ids, actual)
    assert torch.equal(result.candidate_class_ids, candidates)
    assert result.scores.shape == candidates.shape


def test_rerank_candidates_rejects_duplicate_candidates(
    tiny_prompt_repository: CubPromptRepository,
) -> None:
    backend = object.__new__(CLIPBackend)
    backend.device = torch.device("cpu")

    with pytest.raises(ValueError, match="unique"):
        rerank_candidates(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0, 0, 1]]),
            tiny_prompt_repository,
            backend,
        )
